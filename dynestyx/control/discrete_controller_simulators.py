"""Closed-loop control simulator: interleaves simulation, observation, filtering, and control.

Implements the online control loop:

    x_0 ~ p(x_0)
    y_0 | x_0 ~ p(y_0 | x_0, t_0)
    x_hat_{0|0} = FilterUpdate(y_0, t_0)
    u_k, s_{k+1} = control_policy(x_hat_{k|k}, t_k, t_{k+1}, s_k),     k = 0..T-1
        (control_policy always receives the current and next times,
        even if it ignores them; it never receives a key -- a policy
        needing its own randomness must carry it inside s and advance
        it internally, e.g. dynestyx.control.mppi.MPPI. It must return
        a concrete value, not a NumPyro Distribution -- see
        PolicyCallable)
    x_{k+1} | x_k, u_k ~ p(x_{k+1} | x_k, u_k, t_k, t_{k+1}),          k = 0..T-1
    y_{k+1} | x_{k+1}, u_k ~ p(y_{k+1} | x_{k+1}, u_k, t_{k+1}),       k = 0..T-1
    x_hat_{k+1|k+1} = FilterUpdate(x_hat_{k|k}, u_k, y_{k+1}, t_k, t_{k+1}),  k = 0..T-1

`FilterUpdate` is backend-selectable via `filter_config.filter_source`, same
as whole-trajectory filtering (`dynestyx/inference/filters.py`):
`filter_source="cuthbert"` (the default for `PFConfig`/`EnKFConfig`, and one
option for `KFConfig`/`EKFConfig`) drives cuthbert's `Filter.filter_prepare`/
`filter_combine` primitives one step at a time via
`compute_cuthbert_filter_update`
(`dynestyx/inference/integrations/cuthbert/discrete_filter.py`);
`filter_source="cd_dynamax"` (the only option for `UKFConfig`) drives
dynamax's own predict/condition-on primitives one step at a time via
`compute_cd_dynamax_discrete_filter_update`
(`dynestyx/inference/integrations/cd_dynamax/discrete_filter.py`). The two
backends carry genuinely different per-step state (cuthbert: one opaque
belief object; cd_dynamax: a predicted-belief recursion state plus a
separately-returned filtered `(mean, cov)`), so `simulate` has one `_step`
per backend rather than forcing them through a shared representation. The
raw belief cuthbert's step produces is family-specific (e.g. a Kalman-family
state with a `.mean`/`.chol_cov`, or a `ParticleFilterState` with
`.particles`/`.log_weights`), so before it's handed to `control_policy` it's
converted via `filter_state_dist` into a family-agnostic NumPyro
`Distribution` (`MultivariateNormal` for the Gaussian families,
`WeightedParticles` for `PFConfig`); cd_dynamax's step already has a plain
`(mean, cov)` pair, so it builds `dist.MultivariateNormal` directly instead.
Either way, a policy can call `.mean` for a point estimate, or use the full
distribution (e.g. `.sample`) for risk-aware planning.

Important: the control passed to `dynamics.observation_model` for
`y_{k+1}` is `u_k` (the control that drove the transition into `x_{k+1}`),
not a same-index `u_{k+1}`. This differs from `DiscreteTimeSimulator`'s
pre-supplied-trajectory convention, where `ctrl_values[t]` is paired with
both the observation and the outgoing transition at the same index t. That
convention is impossible to satisfy online: `u_{k+1}` is chosen by
`control_policy` from `x_hat_{k+1|k+1}`, which itself depends on having
already observed `y_{k+1}`. See `compute_cuthbert_filter_update`'s docstring
for details.

`DiscreteControlLoopSimulator` computes its own controls online, so unlike
`DiscreteTimeSimulator` it is driven with `predict_times` only -- do not
pass `ctrl_times`/`ctrl_values` to `dsx.sample` (simulator handlers are
generation-only and reject `obs_times`/`obs_values`; `ctrl_values` is
rejected here too, since it would conflict with online control).
"""

import dataclasses
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Real
from numpyro.distributions import Distribution

from dynestyx.inference.configs.filter import BaseFilterConfig
from dynestyx.inference.filters import _default_filter_config
from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (
    build_dynamax_filter,
    compute_cd_dynamax_discrete_filter_update,
)
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


def filter_state_mean(state) -> Array:
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


def filter_state_dist(state) -> Distribution:
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
    `KFConfig`/`EKFConfig`/`EnKFConfig`/`UKFConfig`, `WeightedParticles` for
    `PFConfig` (see module docstring and `filter_state_dist`); use `x_hat.mean` for a
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


@dataclasses.dataclass
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
    filtered_states_mean: Array | None = None
    policy_states: PyTree | None = None


class DiscreteControlLoopSimulator(BaseSimulator):
    r"""Closed-loop simulator: simulate, observe, filter, and decide controls online.

    Unlike `DiscreteTimeSimulator`, which requires the entire control
    trajectory as a pre-supplied `ctrl_values` array, `DiscreteControlLoopSimulator`
    computes each $u_k$ online from the filtered belief $\hat x_{k|k}$ via
    `control_policy`. See the module docstring for the full loop equations
    and the control-index convention used for `dynamics.observation_model`.

    Attributes:
        control_policy: Control policy $\pi$; see `PolicyCallable`. Its initial
            state $s_0$ is exactly `simulate`'s `initial_policy_state` argument
            (default `None`, for a stateless policy) -- `control_policy` is
            never introspected for an `initial_state()` method; a stateful
            policy's initial state must always be passed explicitly.
        filter_config: Selects the filtering algorithm
            (`KFConfig`/`EKFConfig`/`EnKFConfig`/`PFConfig`/`UKFConfig`) and,
            via its `filter_source` field, the backend (`"cuthbert"` or
            `"cd_dynamax"`; `UKFConfig` is cd_dynamax-only, cuthbert has no
            UKF implementation). Defaults to `_default_filter_config(dynamics)`
            when `None`. Its `record_filtered_states_mean`/`record_max_elems`
            fields gate whether the `filtered_states_mean` output is
            recorded, exactly as they do for `Filter` (see
            `dynestyx.utils._should_record_field`).
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

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        initial_policy_state: PyTree | None = None,
        **kwargs,
    ) -> ControlledSimulatedResult:

        if dynamics.continuous_time:
            raise ValueError(
                "DiscreteControlLoopSimulator only supports discrete-time models "
                "(see class docstring). Wrap continuous-time state evolution "
                "in a Discretizer first."
            )
        if ctrl_values is not None:
            raise ValueError(
                "DiscreteControlLoopSimulator computes controls online via "
                "`control_policy`; pass ctrl_values to a plain "
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

        key, k_x0, k_y0, k_filt0 = jr.split(rng_key, 4)
        x_0 = dynamics.initial_condition.sample(k_x0)
        y_0 = dynamics.observation_model(x_0, None, times[0]).sample(k_y0)
        s_0 = initial_policy_state

        if filter_config.filter_source == "cuthbert":
            filter_obj, _ = build_cuthbert_filter(
                dynamics, filter_config, key=rng_key, want_parallel=False
            )

            # Give the bootstrap FilterUpdate a non-degenerate t_prev (borrowing
            # the width of the first real interval, matching
            # compute_cuthbert_filter's own dummy-row convention). This step is
            # a genuine no-op transition for every filter family (nothing has
            # happened before t_0), but some backends (e.g. EKF's Taylor
            # linearization) evaluate the transition unconditionally via
            # jnp.where rather than jax.lax.cond, so t_prev==t would construct
            # a zero-width-dt, zero-covariance distribution whose NaN
            # log-density leaks through the gradient even on the discarded
            # branch -- a state-evolution whose covariance scales with dt
            # (e.g. an Euler-Maruyama-discretized SDE) hits this; a
            # fixed-covariance one (e.g. LinearGaussianStateEvolution) does not.
            dt0 = times[1] - times[0] if T > 1 else jnp.asarray(1.0, dtype=times.dtype)
            x_hat_0 = compute_cuthbert_filter_update(
                dynamics,
                filter_obj=filter_obj,
                prev_state=None,
                key=k_filt0,
                y=y_0,
                u=None,
                t=times[0],
                t_prev=times[0] - dt0,
            )

            def _step(carry, t_idx):
                x_prev, x_hat_prev, s_prev, step_key = carry
                step_key, k_trans, k_obs, k_filt = jr.split(step_key, 4)
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

                trans_dist = dynamics.state_evolution(x_prev, u_k, t_now, t_next)
                x_next = trans_dist.sample(k_trans)

                obs_dist = dynamics.observation_model(x_next, u_k, t_next)
                y_next = obs_dist.sample(k_obs)

                x_hat_next = compute_cuthbert_filter_update(
                    dynamics,
                    filter_obj=filter_obj,
                    prev_state=x_hat_prev,
                    key=k_filt,
                    y=y_next,
                    u=u_k,
                    t=t_next,
                    t_prev=t_now,
                )

                new_carry = (x_next, x_hat_next, s_next, step_key)
                outputs = (x_next, x_hat_next, y_next, s_next, u_k)
                return new_carry, outputs

            init_carry = (x_0, x_hat_0, s_0, key)
            _, (xs, x_hats, ys, ss, us) = jax.lax.scan(
                _step, init_carry, jnp.arange(T - 1)
            )

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

        elif filter_config.filter_source == "cd_dynamax":
            step_fn, _ = build_dynamax_filter(dynamics, filter_config)

            # Unlike cuthbert's bootstrap, no non-degenerate-t_prev trick is
            # needed: build_dynamax_filter bakes its prior into step_fn, which
            # conditions directly on y when prev_state is None, and
            # compute_cd_dynamax_discrete_filter_update ignores t/t_prev
            # entirely (cd_dynamax's discrete-time step functions are already
            # time-homogeneous per step).
            mean_0, cov_0 = compute_cd_dynamax_discrete_filter_update(
                dynamics, step_fn, None, y=y_0, u=None
            )

            def _step(carry, t_idx):
                x_prev, mean_prev, cov_prev, s_prev, step_key = carry
                step_key, k_trans, k_obs = jr.split(step_key, 3)
                t_now = times[t_idx]
                t_next = times[t_idx + 1]

                u_k, s_next = self.control_policy(
                    dist.MultivariateNormal(mean_prev, covariance_matrix=cov_prev),
                    t_now,
                    t_next,
                    s_prev,
                )
                if isinstance(u_k, Distribution):
                    raise ValueError(
                        "Returning a distribution is not yet supported, instead "
                        "sample from this distribution inside your policy."
                    )

                trans_dist = dynamics.state_evolution(x_prev, u_k, t_now, t_next)
                x_next = trans_dist.sample(k_trans)

                obs_dist = dynamics.observation_model(x_next, u_k, t_next)
                y_next = obs_dist.sample(k_obs)

                mean_next, cov_next = compute_cd_dynamax_discrete_filter_update(
                    dynamics,
                    step_fn,
                    (mean_prev, cov_prev),
                    y=y_next,
                    u=u_k,
                    t=t_next,
                    t_prev=t_now,
                )

                new_carry = (x_next, mean_next, cov_next, s_next, step_key)
                outputs = (x_next, mean_next, y_next, s_next, u_k)
                return new_carry, outputs

            init_carry = (x_0, mean_0, cov_0, s_0, key)
            _, (xs, means, ys, ss, us) = jax.lax.scan(
                _step, init_carry, jnp.arange(T - 1)
            )

            mean_shape = mean_0.shape
            record_mean = _should_record_field(
                filter_config.record_filtered_states_mean,
                (T, *mean_shape),
                filter_config.record_max_elems,
            )
            filtered_states_mean = None
            if record_mean:
                filtered_states_mean_vals = jnp.concatenate(
                    [jnp.expand_dims(mean_0, axis=0), means], axis=0
                )
                filtered_states_mean = _ensure_trailing_dim(
                    jnp.expand_dims(filtered_states_mean_vals, axis=0)
                )

        else:
            raise ValueError(f"Unknown filter source: {filter_config.filter_source}")

        states = jnp.concatenate([jnp.expand_dims(x_0, axis=0), xs], axis=0)
        observations = jnp.concatenate([jnp.expand_dims(y_0, axis=0), ys], axis=0)

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
