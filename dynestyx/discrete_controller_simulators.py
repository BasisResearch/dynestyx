"""Closed-loop control simulator: interleaves simulation, observation, filtering, and control.

Implements the online control loop:

    x_0 ~ p(x_0)
    y_0 | x_0 ~ p(y_0 | x_0, t_0)
    x_hat_{0|0} = FilterUpdate(y_0, t_0)
    u_k, s_{k+1} = control_policy(x_hat_{k|k}, s_k),                  k = 0..T-1
    x_{k+1} | x_k, u_k ~ p(x_{k+1} | x_k, u_k, t_k, t_{k+1}),          k = 0..T-1
    y_{k+1} | x_{k+1}, u_k ~ p(y_{k+1} | x_{k+1}, u_k, t_{k+1}),       k = 0..T-1
    x_hat_{k+1|k+1} = FilterUpdate(x_hat_{k|k}, u_k, y_{k+1}, t_k, t_{k+1}),  k = 0..T-1

`FilterUpdate` is implemented by `compute_cuthbert_filter_update`
(`dynestyx/inference/integrations/cuthbert/discrete_filter.py`), which drives
cuthbert's `Filter.filter_prepare`/`filter_combine` primitives one step at a
time instead of over a whole pre-supplied trajectory. This works for any
filter family exposed there (`KFConfig`, `EKFConfig`, `EnKFConfig`,
`PFConfig`); the belief `x_hat` returned each step is therefore whatever
state type that family produces (e.g. a Kalman-family state with a `.mean`
property, or a `ParticleFilterState` with `.particles`/`.log_weights`) --
see `filter_state_mean` below for a family-agnostic point estimate.

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
pass `ctrl_times`/`ctrl_values` to `dsx.sample` (the shared validation in
`dynestyx/utils.py` requires them together and rejects `ctrl_times` alone,
and `ctrl_values` would conflict with online control in any case).
"""

import dataclasses
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from jax import Array
from jaxtyping import PyTree, Real

from dynestyx.inference.filter_configs import BaseFilterConfig
from dynestyx.inference.filters import _default_filter_config
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    compute_cuthbert_filter_update,
)
from dynestyx.models import DynamicalModel
from dynestyx.simulators import BaseSimulator, _ensure_trailing_dim, _tile_times
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


@runtime_checkable
class PolicyCallable(Protocol):
    r"""Structural protocol for a control policy $\pi$.

    $$u_k, s_{k+1} = \pi(\hat x_{k|k}, s_k)$$

    `x_hat` is whatever belief state the chosen `filter_config` family
    produces (see module docstring); use `filter_state_mean` for a
    family-agnostic point estimate. Any plain callable matching this
    signature works, including an `equinox.Module` with a matching
    `__call__` (e.g. a learned neural policy) or a plain Python function
    (e.g. an LQR gain lookup).
    """

    def __call__(
        self, x_hat: Any, s: PyTree
    ) -> tuple[Real[Array, " control_dim"], PyTree]:
        raise NotImplementedError()


@dataclasses.dataclass
class DiscreteControlLoopSimulator(BaseSimulator):
    r"""Closed-loop simulator: simulate, observe, filter, and decide controls online.

    Unlike `DiscreteTimeSimulator`, which requires the entire control
    trajectory as a pre-supplied `ctrl_values` array, `DiscreteControlLoopSimulator`
    computes each $u_k$ online from the filtered belief $\hat x_{k|k}$ via
    `control_policy`. See the module docstring for the full loop equations
    and the control-index convention used for `dynamics.observation_model`.

    Attributes:
        control_policy: Control policy $\pi$; see `PolicyCallable`.
        policy_state_init: Initial policy state $s_0$ (any PyTree).
        filter_config: Selects the filtering algorithm
            (`KFConfig`/`EKFConfig`/`EnKFConfig`/`PFConfig`). Defaults to
            `_default_filter_config(dynamics)` when `None`. Its
            `record_filtered_states_mean`/`record_max_elems` fields gate
            whether the `filtered_states_mean` output is recorded, exactly
            as they do for `Filter` (see `dynestyx.utils._should_record_field`).
        n_simulations: Currently only `1` is supported.
    """

    control_policy: PolicyCallable
    policy_state_init: PyTree
    filter_config: BaseFilterConfig | None = None
    n_simulations: int = 1

    def _simulate(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        _obs_values_filled=None,
        _obs_mask=None,
        _obs_has_missing=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, Array]:
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
        if obs_values is not None:
            raise ValueError(
                "DiscreteControlLoopSimulator does not support conditioning on "
                "obs_values yet; it only supports forward rollout."
            )
        if self.n_simulations != 1:
            raise NotImplementedError(
                "DiscreteControlLoopSimulator does not yet support n_simulations > 1."
            )

        times = obs_times if obs_times is not None else predict_times
        if times is None:
            raise ValueError("obs_times or predict_times must be provided")
        T = len(times)
        if T < 1:
            raise ValueError("times must contain at least one timepoint")

        filter_config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )

        key = numpyro.prng_key()
        if key is None:
            raise ValueError(
                "DiscreteControlLoopSimulator requires a PRNG key (run inside a "
                "seeded context, e.g. numpyro.handlers.seed)."
            )
        key, k_x0, k_y0, k_filt0 = jr.split(key, 4)

        x_0 = dynamics.initial_condition.sample(k_x0)
        y_0 = dynamics.observation_model(x_0, None, times[0]).sample(k_y0)
        # Give the bootstrap FilterUpdate a non-degenerate t_prev (borrowing the
        # width of the first real interval, matching compute_cuthbert_filter's
        # own dummy-row convention). This step is a genuine no-op transition
        # for every filter family (nothing has happened before t_0), but some
        # backends (e.g. EKF's Taylor linearization) evaluate the transition
        # unconditionally via jnp.where rather than jax.lax.cond, so t_prev==t
        # would construct a zero-width-dt, zero-covariance distribution whose
        # NaN log-density leaks through the gradient even on the discarded
        # branch -- a state-evolution whose covariance scales with dt (e.g. an
        # Euler-Maruyama-discretized SDE) hits this; a fixed-covariance one
        # (e.g. LinearGaussianStateEvolution) does not.
        dt0 = times[1] - times[0] if T > 1 else jnp.asarray(1.0, dtype=times.dtype)
        x_hat_0 = compute_cuthbert_filter_update(
            dynamics,
            filter_config,
            None,
            k_filt0,
            y=y_0,
            u=None,
            t=times[0],
            t_prev=times[0] - dt0,
        )
        s_0 = self.policy_state_init

        def _step(carry, t_idx):
            x_prev, x_hat_prev, s_prev, step_key = carry
            step_key, k_trans, k_obs, k_filt = jr.split(step_key, 4)
            t_now = times[t_idx]
            t_next = times[t_idx + 1]

            u_k, s_next = self.control_policy(x_hat_prev, s_prev)

            trans_dist = dynamics.state_evolution(x_prev, u_k, t_now, t_next)
            x_next = trans_dist.sample(k_trans)

            obs_dist = dynamics.observation_model(x_next, u_k, t_next)
            y_next = obs_dist.sample(k_obs)

            x_hat_next = compute_cuthbert_filter_update(
                dynamics,
                filter_config,
                x_hat_prev,
                k_filt,
                y=y_next,
                u=u_k,
                t=t_next,
                t_prev=t_now,
            )

            new_carry = (x_next, x_hat_next, s_next, step_key)
            outputs = (x_next, x_hat_next, y_next, s_next, u_k)
            return new_carry, outputs

        init_carry = (x_0, x_hat_0, s_0, key)
        _, (xs, x_hats, ys, ss, us) = jax.lax.scan(_step, init_carry, jnp.arange(T - 1))

        states = jnp.concatenate([jnp.expand_dims(x_0, axis=0), xs], axis=0)
        observations = jnp.concatenate([jnp.expand_dims(y_0, axis=0), ys], axis=0)

        result = {
            "times": _tile_times(times, 1),
            "states": _ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
            "observations": _ensure_trailing_dim(jnp.expand_dims(observations, axis=0)),
            "controls": _ensure_trailing_dim(jnp.expand_dims(us, axis=0)),
        }

        mean_shape = filter_state_mean(x_hat_0).shape
        record_mean = _should_record_field(
            filter_config.record_filtered_states_mean,
            (T, *mean_shape),
            filter_config.record_max_elems,
        )
        if record_mean:
            filtered_states_mean = jnp.concatenate(
                [
                    jnp.expand_dims(filter_state_mean(x_hat_0), axis=0),
                    filter_state_mean(x_hats),
                ],
                axis=0,
            )
            result["filtered_states_mean"] = _ensure_trailing_dim(
                jnp.expand_dims(filtered_states_mean, axis=0)
            )

        if s_0 is not None:
            # A stateless policy (policy_state_init=None) has nothing to
            # record; jnp.expand_dims can't be applied to None directly, and
            # there is no meaningful "policy_states" trajectory to report.
            result["policy_states"] = jax.tree_util.tree_map(
                lambda leaf: jnp.expand_dims(leaf, axis=0), ss
            )
        return result


__all__ = ["DiscreteControlLoopSimulator", "PolicyCallable", "filter_state_mean"]
