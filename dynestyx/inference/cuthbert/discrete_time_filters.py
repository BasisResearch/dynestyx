from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
from cuthbert import filter as cuthbert_filter
from cuthbert.gaussian import taylor
from cuthbert.smc import particle_filter

from dynestyx.cuthbert_patches import systematic_resampling
from dynestyx.dynamical_models import Context, DynamicalModel
from dynestyx.utils import _get_controls, _validate_control_dim

_DISCRETE_FILTER_TYPES: list[str] = ["default", "taylor_kf", "pf"]


class _CuthbertInputs(NamedTuple):
    """Model inputs pytree for cuthbert; leading time dim must be T+1."""

    y: jax.Array  # (T+1, emission_dim)
    u: jax.Array  # (T+1, control_dim) or (T+1, 0)
    u_prev: jax.Array  # (T+1, control_dim) or (T+1, 0)
    time: jax.Array  # (T+1,)
    time_prev: jax.Array  # (T+1,)


def _filter_discrete_time(
    name: str,
    filter_type: str,
    dynamics: DynamicalModel,
    context: Context,
    key: jax.Array | None = None,
    filter_kwargs: dict | None = None,
):
    """
    Discrete-time marginal likelihood via cuthbert.
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    obs_traj = context.observations
    if obs_traj.values is None:
        return
    if isinstance(obs_traj.values, dict):
        raise ValueError("obs_traj.values must be an Array, not a dict")

    ys = obs_traj.values
    T1 = int(ys.shape[0])  # this is T+1 in cuthbert's convention
    if T1 == 0:
        return

    # Time axis (scalar at each step after slicing by cuthbert.filter)
    if obs_traj.times is None:
        times = jnp.arange(T1, dtype=jnp.float32)
    else:
        times = jnp.asarray(obs_traj.times)

    # Align controls (if any) to observation times
    _, ctrl_values = _get_controls(context, times)
    _validate_control_dim(dynamics, ctrl_values)

    if ctrl_values is None:
        ctrl_values = jnp.zeros((T1, 0), dtype=ys.dtype)

    dt0 = times[1] - times[0]

    time_prev = jnp.concatenate([times[:1] - dt0, times[:-1]], axis=0)

    u_prev = jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0)

    key = key if key is not None else numpyro.prng_key()

    cuthbert_inputs = _CuthbertInputs(
        y=ys, u=ctrl_values, u_prev=u_prev, time=times, time_prev=time_prev
    )

    if filter_type.lower() in ["taylor_kf", "default"]:
        filter_obj = _cuthbert_filter_taylor_kf(dynamics, filter_kwargs)
    elif filter_type.lower() == "pf":
        filter_obj = _cuthbert_filter_pf(dynamics, filter_kwargs)
    else:
        raise ValueError(
            f"Invalid filter type: {filter_type}. Valid types: {_DISCRETE_FILTER_TYPES}"
        )

    states = cuthbert_filter(filter_obj, cuthbert_inputs, parallel=False, key=key)

    marginal_loglik = states.log_normalizing_constant[-1]

    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)


def _cuthbert_filter_pf(dynamics: DynamicalModel, filter_kwargs: dict | None = None):
    if filter_kwargs is None:
        filter_kwargs = {}

    def init_sample(key, mi: _CuthbertInputs):
        return dynamics.initial_condition.sample(key)

    def propagate_sample(key, x_prev, mi: _CuthbertInputs):
        # TODO: Resolve these types later.
        dist = dynamics.state_evolution(x_prev, mi.u_prev, mi.time_prev, mi.time)  # type: ignore
        return dist.sample(key)  # type: ignore

    def log_potential(x_prev, x, mi: _CuthbertInputs):
        edist = dynamics.observation_model(x, mi.u, mi.time)
        return jnp.asarray(edist.log_prob(mi.y)).sum()

    ess_threshold = filter_kwargs.get("ess_threshold", 0.7)

    pf = particle_filter.build_filter(
        init_sample=init_sample,  # type: ignore
        propagate_sample=propagate_sample,  # type: ignore
        log_potential=log_potential,  # type: ignore
        n_filter_particles=int(filter_kwargs.get("n_filter_particles", 1_000)),
        resampling_fn=systematic_resampling.resampling,  # type: ignore
        ess_threshold=ess_threshold,
    )
    return pf


def _cuthbert_filter_taylor_kf(
    dynamics: DynamicalModel, filter_kwargs: dict | None = None
):
    if filter_kwargs is None:
        filter_kwargs = {}

    rtol = filter_kwargs.get("rtol", None)

    def get_init_log_density(mi: _CuthbertInputs):
        dist0 = dynamics.initial_condition

        def init_log_density(x):
            return jnp.asarray(dist0.log_prob(x)).sum()

        x0_lin = dist0.mean
        return init_log_density, x0_lin

    def get_dynamics_log_density(
        state: taylor.LinearizedKalmanFilterState, mi: _CuthbertInputs
    ):
        # log p(x_t | x_{t-1})
        def dynamics_log_density(x_prev, x):
            dist = dynamics.state_evolution(x_prev, mi.u_prev, mi.time_prev, mi.time)
            return jnp.asarray(dist.log_prob(x)).sum()

        # Linearize around previous filtered mean.
        x_prev_lin = state.mean

        # A decent guess for the x_t linearization point is the conditional mean at x_prev_lin (if available).
        dist_at_lin = dynamics.state_evolution(  # type: ignore
            x_prev_lin, mi.u_prev, mi.time_prev, mi.time
        )
        try:
            x_lin = dist_at_lin.mean  # type: ignore
        except Exception:
            raise ValueError(
                "dist_at_lin.mean is not available. Linearized Kalman filter requires a mean-able distribution."
            )

        return dynamics_log_density, x_prev_lin, x_lin

    def get_observation_func(
        state: taylor.LinearizedKalmanFilterState, mi: _CuthbertInputs
    ):
        def log_potential(x):
            edist = dynamics.observation_model(x, mi.u, mi.time)
            return jnp.asarray(edist.log_prob(mi.y)).sum()

        return log_potential, state.mean

    kf = taylor.build_filter(
        get_init_log_density,  # type: ignore
        get_dynamics_log_density,  # type: ignore
        get_observation_func,  # type: ignore
        associative=False,
        rtol=rtol,
        ignore_nan_dims=True,
    )

    return kf
