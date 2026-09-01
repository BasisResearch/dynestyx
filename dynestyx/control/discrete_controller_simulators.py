"""Closed-loop simulation for controlled discrete-time dynamical models."""

from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Real
from numpyro.distributions import Distribution

from dynestyx.inference.configs.filter import BaseFilterConfig
from dynestyx.inference.filters import _default_filter_config
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    build_cuthbert_filter,
    compute_cuthbert_filter_update,
)
from dynestyx.inference.integrations.utils import WeightedParticles
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


def filter_state_dist(state: Any) -> Distribution:
    """Full-belief NumPyro distribution for a cuthbert filter state, any family.

    Kalman-family states (`KFConfig`, `EKFConfig`, `EnKFConfig`) expose
    `.mean`/`.chol_cov`, giving an exact `MultivariateNormal`. `PFConfig`
    states have no such property -- their belief is a weighted particle
    cloud (`.particles`, `.log_weights`), represented via `WeightedParticles`
    (dynestyx's own `Distribution`; NumPyro has no built-in equivalent).
    Unlike `filter_state_mean`, this does not broadcast over a leading
    time/batch axis -- call it once per (unbatched) state.
    """
    if hasattr(state, "chol_cov"):
        return dist.MultivariateNormal(state.mean, scale_tril=state.chol_cov)
    if hasattr(state, "particles") and hasattr(state, "log_weights"):
        log_weights = jax.nn.log_softmax(state.log_weights, axis=-1)
        return WeightedParticles(state.particles, log_weights)
    raise TypeError(
        f"Cannot build a distribution for filter state of type {type(state).__name__}"
    )


@runtime_checkable
class PolicyCallable(Protocol):
    r"""Structural protocol for a control policy $\pi$.

    $$u_k, s_{k+1} = \pi(\hat x_{k|k}, t_k, t_{k+1}, s_k)$$

    `x_hat` is a NumPyro `Distribution` -- `MultivariateNormal` for
    `KFConfig`/`EKFConfig`/`EnKFConfig`, `WeightedParticles` for `PFConfig`
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
    """

    # control_time = time - 1 (no control is chosen after the final state).
    controls: Real[Array, "n_simulations control_time control_dim"] | None = None
    filtered_states_mean: Real[Array, "n_simulations time state_dim"] | None = None
    policy_states: PyTree | None = None


class DiscreteControlLoopSimulator(BaseSimulator):
    r"""Closed-loop simulator: simulate, observe, filter, and decide controls online.

    Unlike `DiscreteTimeSimulator`, which requires the entire control
    trajectory as a pre-supplied `ctrl_values` array, `DiscreteControlLoopSimulator`
    computes each $u_k$ online from the filtered belief $\hat x_{k|k}$ via
    `control_policy`. See the closed-loop control API page for the full loop
    equations and the control-index convention used by
    `dynamics.observation_model`.

    The online loop uses $u_k$ for both the transition into $x_{k+1}$ and the
    observation $y_{k+1}$. This control-observation alignment is temporary;
    [Issue #312](https://github.com/BasisResearch/dynestyx/issues/312) tracks
    aligning it with the regular simulator convention and requiring controlled
    `DynamicalModel` observation models to follow that convention. The one-step
    filter update currently uses Cuthbert and supports `KFConfig`, `EKFConfig`,
    `EnKFConfig`, and `PFConfig`. Plated controlled simulation is not yet
    supported; see
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
            States, observations, controls, filter means, and policy states.

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
        rollout_key, initial_state_key, initial_observation_key, default_filter_key = (
            jr.split(rng_key, 4)
        )
        online_filter_key = (
            filter_config.crn_seed
            if filter_config.crn_seed is not None
            else default_filter_key
        )
        online_filter_key, initial_filter_update_key = jr.split(online_filter_key)
        filter_obj, _ = build_cuthbert_filter(
            dynamics, filter_config, key=online_filter_key, want_parallel=False
        )

        x_0 = dynamics.initial_condition.sample(initial_state_key)
        y_0 = dynamics.observation_model(x_0, None, times[0]).sample(
            initial_observation_key
        )
        # This first filter update conditions the initial-state prior on y_0;
        # it does not perform a state transition. t_prev is therefore a dummy
        # value, but it must be earlier than t_0 because some filter backends
        # still evaluate the unused transition. Reusing the first interval's
        # width avoids zero-duration transition covariances and NaN gradients.
        dt0 = times[1] - times[0] if T > 1 else jnp.asarray(1.0, dtype=times.dtype)
        x_hat_0 = compute_cuthbert_filter_update(
            dynamics,
            filter_obj=filter_obj,
            prev_state=None,
            key=initial_filter_update_key,
            y=y_0,
            u=None,
            t=times[0],
            t_prev=times[0] - dt0,
        )
        s_0 = initial_policy_state

        def _step(carry, t_idx):
            x_prev, x_hat_prev, s_prev, rollout_key, online_filter_key = carry
            rollout_key, transition_key, observation_key = jr.split(rollout_key, 3)
            online_filter_key, filter_update_key = jr.split(online_filter_key)
            t_now = times[t_idx]
            t_next = times[t_idx + 1]

            u_k, s_next = self.control_policy(
                filter_state_dist(x_hat_prev), t_now, t_next, s_prev
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
        observations = jnp.concatenate([jnp.expand_dims(y_0, axis=0), ys], axis=0)

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
