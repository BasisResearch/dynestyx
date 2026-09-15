"""Closed-loop simulation for controlled discrete-time dynamical models."""

from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Real
from numpyro.distributions import Distribution

from dynestyx.inference.configs.filter import BaseFilterConfig, PFConfig
from dynestyx.inference.filters import _default_filter_config
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    CuthbertInputs,
    build_cuthbert_filter,
    compute_cuthbert_filter_update,
)
from dynestyx.inference.utils.distribution_utils import (
    _cholesky_state_sequence_to_dists,
)
from dynestyx.models import DynamicalModel
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import _ensure_trailing_dim, _tile_times
from dynestyx.types import SimulatedResult
from dynestyx.utils import _should_record_field


def filter_state_mean(state: Any) -> Real[Array, "..."]:
    """Point-estimate summary of a cuthbert filter state, any family.

    Kalman-family states (`KFConfig`, `EKFConfig`, `EnKFConfig`) expose a
    `.mean` property directly. `PFConfig` states (`ParticleFilterState`) have
    no such property -- they represent the belief as a weighted particle
    cloud (`.particles`, `.log_weights`), so the point estimate is the
    weighted mean instead. Broadcasts over any leading batch/time axis, so it
    works on both a single belief and a whole scanned-out sequence of them.
    """
    if hasattr(state, "mean"):
        return state.mean
    if hasattr(state, "particles") and hasattr(state, "log_weights"):
        weights = jax.nn.softmax(state.log_weights, axis=-1)
        return jnp.sum(weights[..., None] * state.particles, axis=-2)
    raise TypeError(f"Cannot summarize filter state of type {type(state).__name__}")


def filter_state_dist(state: Any, filter_config: BaseFilterConfig) -> Distribution:
    """Full-belief NumPyro distribution for a filter state.

    Only `filter_source="cuthbert"` is supported today, matching
    `DiscreteControlLoopSimulator`'s own restriction. Converts the filter state to a NumPyro `Distribution`
    in the same way that `ConditionedResult.dists` does for the recorded filtered states.

    The shared conversion is time-indexed, so the state is given a leading axis of
    length one and the single distribution unwrapped.

    Args:
        state: A single, unbatched filter state produced by `filter_config`.
        filter_config: The config that produced `state`. Selects the backend and
            carries `recorded_filtered_states_cov_jitter` for ensemble states.

    Returns:
        The belief as a NumPyro `Distribution`.

    Raises:
        ValueError: If `filter_config.filter_source` is not `"cuthbert"`.
    """

    if filter_config.filter_source == "cuthbert":
        # Give the time-indexed conversion a leading axis of one
        with_time_axis = SimpleNamespace(
            **{
                name: jnp.asarray(getattr(state, name))[None]
                for name in ("ensemble", "mean", "chol_cov", "particles", "log_weights")
                if hasattr(state, name)
            }
        )
        return _cholesky_state_sequence_to_dists(
            with_time_axis,
            particle_mode=isinstance(filter_config, PFConfig),
            covariance_jitter=getattr(
                filter_config, "recorded_filtered_states_cov_jitter", 0.0
            ),
        )[0]
    else:
        raise ValueError(
            "filter_state_dist currently only supports filter_source='cuthbert' only, got "
            f"{filter_config.filter_source!r}."
        )


@runtime_checkable
class PolicyCallable(Protocol):
    r"""Structural protocol for a control policy $\pi$.

    $$u_k, s_{k+1} = \pi(\hat p_k, t_k, t_{k+1}, s_k)$$

    `x_hat` represents the current belief $\hat p_k$. On the first policy
    call it is the model's initial-state distribution $p_0$; after each
    generated observation it is the corresponding filtered distribution.
    It is a NumPyro `Distribution` -- `MultivariateNormal` for
    `KFConfig`/`EKFConfig`, `WeightedParticles` for `PFConfig`, and for
    `EnKFConfig` either of `MultivariateNormal` or, once the ensemble is rank
    deficient (`n_particles - 1 < state_dim`), `LowRankMultivariateNormal`
    (see `filter_state_dist`); use `x_hat.mean` for a
    family-agnostic point estimate, or the distribution itself for
    uncertainty-aware planning. `t_now`/`t_next` are the current and next
    times -- always passed, even to a policy that ignores them, so that a
    policy needing genuine time-dependence (e.g. `dynestyx.control.mppi.MPPI`,
    which plans forward from `t_now`) doesn't need special-casing. Any plain
    callable matching this signature works, including an `equinox.Module`
    with a matching `__call__` (e.g. a learned neural policy) or a plain
    Python function (e.g. an LQR gain lookup).

    `control_policy` never receives a PRNG key and must return a concrete
    value, not a NumPyro `Distribution` (returning one raises a `ValueError`
    -- not yet supported). A stochastic policy instead carries any
    randomness it needs (e.g. MPPI's exploration noise) inside `s`,
    splitting/advancing it internally on every call and sampling from its
    own distributions itself before returning a value. The policy owns and
    seeds this randomness entirely by itself -- see
    `dynestyx.control.mppi.MPPI`'s `seed` attribute for the pattern.
    """

    def __call__(
        self,
        x_hat: Distribution,
        t_now: Real[Array, ""],
        t_next: Real[Array, ""],
        s: PyTree,
    ) -> tuple[Real[Array, " control_dim"], PyTree]:
        raise NotImplementedError()


class ControlledSimulatedResult(SimulatedResult):
    """`SimulatedResult` extended with the control loop's extra outputs.

    Registered as deterministic sites the same generic way as
    `SimulatedResult`'s own fields (`dynestyx.simulation.utils.
    _register_simulated_result_sites` iterates every dataclass field and
    skips `None` values) -- so the existing recording-gating logic just
    means passing `None` for a field instead of conditionally omitting a
    dict key, as the old (pre-refactor) version of this class did.

    Closed-loop simulation always uses the previous-transition convention:
    `times`, `states`, and (when requested) `filtered_states_mean` have length
    $T$, while `observations`, `controls`, and `policy_states` have length
    $T-1$. No initial observation $y_0$ is generated.
    """

    # control_time = time - 1 (no control is chosen after the final state).
    controls: Real[Array, "n_simulations control_time control_dim"] | None = None
    filtered_states_mean: Real[Array, "n_simulations time state_dim"] | None = None
    policy_states: PyTree | None = None


class DiscreteControlLoopSimulator(BaseSimulator):
    r"""Closed-loop simulator: simulate, observe, filter, and decide controls online.

    Unlike `DiscreteTimeSimulator`, which requires the entire control
    trajectory as a pre-supplied `ctrl_values` array, `DiscreteControlLoopSimulator`
    computes each $u_k$ online from the current belief $\hat p_k$ via
    `control_policy`:

    $$
    x_0 \sim p_0, \qquad \hat p_0 = p_0,
    $$

    followed for $k=0,\ldots,T-2$ by

    $$
    \begin{aligned}
    (u_k,s_{k+1}) &= \pi(\hat p_k,t_k,t_{k+1},s_k), \\
    x_{k+1} &\sim p(x_{k+1}\mid x_k,u_k,t_k,t_{k+1}), \\
    y_{k+1} &\sim p(y_{k+1}\mid x_{k+1},u_k,t_{k+1}), \\
    \hat p_{k+1} &= \operatorname{FilterUpdate}(\hat p_k,u_k,y_{k+1}).
    \end{aligned}
    $$

    Closed-loop simulation therefore always uses the previous-transition
    convention, independently of `dynamics.observation_control_alignment`.
    It never generates $y_0$: states and times have length $T$, while
    observations and controls have length $T-1$. This avoids requiring an
    undefined pre-initial control for a control-dependent observation model.

    The one-step filter update currently uses Cuthbert and supports `KFConfig`,
    `EKFConfig`, `EnKFConfig`, and `PFConfig`. Plated controlled simulation is
    not yet supported; see
    [Issue #318](https://github.com/BasisResearch/dynestyx/issues/318).

    Attributes:
        control_policy: Control policy $\pi$; see `PolicyCallable`. Its initial
            state $s_0$ is exactly `simulate`'s `initial_policy_state` argument
            (default `None`, for a stateless policy) -- `control_policy` is
            never introspected for an `initial_state()` method; a stateful
            policy's initial state must always be passed explicitly.
        filter_config: Selects the filtering algorithm
            (`KFConfig`/`EKFConfig`/`EnKFConfig`/`PFConfig`). Defaults to
            `_default_filter_config(dynamics)` when `None`. The online one-step
            update currently requires `filter_source="cuthbert"`. Its
            `record_filtered_states_mean`/`record_max_elems` fields gate
            whether the `filtered_states_mean` output is recorded, exactly
            as they do for `Filter` (see `dynestyx.utils._should_record_field`).
        n_simulations: Currently only `1` is supported.
    """

    def __init__(
        self,
        *,
        control_policy: PolicyCallable,
        filter_config: BaseFilterConfig | None = None,
        n_simulations: int = 1,
    ) -> None:
        super().__init__(n_simulations=n_simulations)
        self.control_policy = control_policy
        self.filter_config = filter_config

    def _validate_plate_support(self) -> None:
        raise NotImplementedError(
            "DiscreteControlLoopSimulator does not yet support dsx.plate. "
            "Run one controlled model at a time."
        )

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        ctrl_times: Real[Array, " ctrl_time"] | None = None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None = None,
        predict_times: Real[Array, " predict_time"] | None = None,
        initial_policy_state: PyTree | None = None,
        **kwargs: Any,
    ) -> ControlledSimulatedResult:
        """Simulate one online controlled trajectory.

        Args:
            dynamics: Discrete-time dynamical model.
            rng_key: Root key for environment and fallback filter randomness.
            ctrl_times: Unsupported because controls are selected online.
            ctrl_values: Unsupported because controls are selected online.
            predict_times: Strictly increasing simulation times.
            initial_policy_state: Initial state passed to `control_policy`.
            **kwargs: Additional shared simulator-handler metadata, ignored here.

        Returns:
            States and beliefs on all `predict_times`, plus observations,
            controls, and policy states for the subsequent $T-1$ transitions.

        Raises:
            ValueError: If inputs are incompatible with online discrete control.
            NotImplementedError: If the requested simulation mode is unsupported.
        """

        del kwargs
        if dynamics.continuous_time:
            raise ValueError(
                "DiscreteControlLoopSimulator only supports discrete-time models "
                "(see class docstring). Wrap continuous-time state evolution "
                "in a Discretizer first."
            )
        if ctrl_times is not None or ctrl_values is not None:
            raise ValueError(
                "DiscreteControlLoopSimulator computes controls online via "
                "`control_policy`; do not pass ctrl_times or ctrl_values. Use a plain "
                "Simulator/DiscreteTimeSimulator instead if you want "
                "open-loop control."
            )
        if self.n_simulations != 1:
            raise NotImplementedError(
                "DiscreteControlLoopSimulator does not yet support n_simulations > 1."
            )

        times = predict_times
        if times is None:
            raise ValueError("predict_times must be provided")
        T = len(times)
        if T < 1:
            raise ValueError("times must contain at least one timepoint")

        filter_config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )
        if filter_config.filter_source != "cuthbert":
            # TODO: lift this restriction once cd-dynamax filter sources support
            # online one-step updates -- tracked in
            # https://github.com/BasisResearch/dynestyx/pull/314.
            raise ValueError(
                "DiscreteControlLoopSimulator requires filter_source='cuthbert' "
                "because online one-step updates are not available for "
                f"filter_source={filter_config.filter_source!r}."
            )
        rollout_key, initial_state_key, default_filter_key = jr.split(rng_key, 3)
        online_filter_key = (
            filter_config.crn_seed
            if filter_config.crn_seed is not None
            else default_filter_key
        )
        online_filter_key, initial_filter_state_key = jr.split(online_filter_key)
        filter_obj, _ = build_cuthbert_filter(
            dynamics, filter_config, key=online_filter_key, want_parallel=False
        )

        x_0 = dynamics.initial_condition.sample(initial_state_key)
        initial_dtype = jnp.result_type(jnp.asarray(x_0), times)
        zero_control = jnp.zeros((dynamics.control_dim,), dtype=initial_dtype)
        initial_filter_inputs = CuthbertInputs(
            y=jnp.zeros((dynamics.observation_dim,), dtype=initial_dtype),
            u=zero_control,
            u_prev=zero_control,
            time=times[0],
            time_prev=times[0],
            is_first_step=jnp.asarray(False),
        )
        # The first policy acts on the model prior. There is deliberately no
        # synthetic y_0: the first observation follows the transition driven by
        # u_0, exactly like every subsequent closed-loop observation.
        x_hat_0 = filter_obj.init_prepare(
            initial_filter_inputs, key=initial_filter_state_key
        )
        s_0 = initial_policy_state

        def _step(carry, t_idx):
            x_prev, x_hat_prev, s_prev, rollout_key, online_filter_key = carry
            rollout_key, transition_key, observation_key = jr.split(rollout_key, 3)
            online_filter_key, filter_update_key = jr.split(online_filter_key)
            t_now = times[t_idx]
            t_next = times[t_idx + 1]

            u_k, s_next = self.control_policy(
                filter_state_dist(x_hat_prev, filter_config),
                t_now,
                t_next,
                s_prev,
            )
            if isinstance(u_k, Distribution):
                raise ValueError(
                    "Returning a distribution is not yet supported, instead "
                    "sample from this distribution inside your policy."
                )
            u_k = jnp.asarray(u_k)
            expected_control_shape = (dynamics.control_dim,)
            if u_k.shape != expected_control_shape:
                raise ValueError(
                    "control_policy must return one control vector with shape "
                    f"{expected_control_shape}; got {u_k.shape}."
                )

            trans_dist = dynamics.state_evolution(x_prev, u_k, t_now, t_next)
            x_next = trans_dist.sample(transition_key)

            obs_dist = dynamics.observation_model(x_next, u_k, t_next)
            y_next = obs_dist.sample(observation_key)

            x_hat_next = compute_cuthbert_filter_update(
                dynamics,
                filter_obj=filter_obj,
                prev_state=x_hat_prev,
                key=filter_update_key,
                y=y_next,
                u=u_k,
                t=t_next,
                t_prev=t_now,
            )

            new_carry = (
                x_next,
                x_hat_next,
                s_next,
                rollout_key,
                online_filter_key,
            )
            outputs = (x_next, x_hat_next, y_next, s_next, u_k)
            return new_carry, outputs

        init_carry = (x_0, x_hat_0, s_0, rollout_key, online_filter_key)
        _, (xs, x_hats, ys, ss, us) = jax.lax.scan(_step, init_carry, jnp.arange(T - 1))

        states = jnp.concatenate([jnp.expand_dims(x_0, axis=0), xs], axis=0)
        observations = ys

        mean_shape = filter_state_mean(x_hat_0).shape
        record_mean = _should_record_field(
            filter_config.record_filtered_states_mean,
            (T, *mean_shape),
            filter_config.record_max_elems,
        )
        filtered_states_mean = None
        if record_mean:
            filtered_states_mean_vals = jnp.concatenate(
                [
                    jnp.expand_dims(filter_state_mean(x_hat_0), axis=0),
                    filter_state_mean(x_hats),
                ],
                axis=0,
            )
            filtered_states_mean = _ensure_trailing_dim(
                jnp.expand_dims(filtered_states_mean_vals, axis=0)
            )

        policy_states = None
        if s_0 is not None:
            # A stateless policy (no initial_policy_state given) has nothing
            # to record; jnp.expand_dims can't be applied to None directly,
            # and there is no meaningful "policy_states" trajectory to report.
            policy_states = jax.tree_util.tree_map(
                lambda leaf: jnp.expand_dims(leaf, axis=0), ss
            )

        return ControlledSimulatedResult(
            times=_tile_times(times, 1),
            x_0=jnp.expand_dims(x_0, axis=0),
            states=_ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
            observations=_ensure_trailing_dim(jnp.expand_dims(observations, axis=0)),
            controls=_ensure_trailing_dim(jnp.expand_dims(us, axis=0)),
            filtered_states_mean=filtered_states_mean,
            policy_states=policy_states,
        )


__all__ = [
    "ControlledSimulatedResult",
    "DiscreteControlLoopSimulator",
    "PolicyCallable",
    "filter_state_dist",
    "filter_state_mean",
]
