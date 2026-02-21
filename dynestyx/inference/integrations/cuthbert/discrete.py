from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro

from cuthbert import filter as cuthbert_filter
from cuthbert.gaussian import taylor
from cuthbert.smc import particle_filter
from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    EKFConfig,
    EnKFConfig,
    PFConfig,
    config_to_record_kwargs,
)
from dynestyx.inference.integrations.cuthbert.patches import systematic_resampling
from dynestyx.models import DynamicalModel
from dynestyx.utils import (
    _should_record_field,
    _validate_control_dim,
    _validate_controls,
)


class CuthbertInputs(NamedTuple):
    """Model inputs pytree for cuthbert; leading time dim must be T+1."""

    y: jax.Array  # (T+1, emission_dim)
    u: jax.Array  # (T+1, control_dim) or (T+1, 0)
    u_prev: jax.Array  # (T+1, control_dim) or (T+1, 0)
    time: jax.Array  # (T+1,)
    time_prev: jax.Array  # (T+1,)


def _config_to_filter_kwargs(config: BaseFilterConfig) -> dict:
    """Build filter_kwargs dict from config dataclass."""
    kwargs = dict(config.extra_filter_kwargs)
    if isinstance(config, PFConfig):
        kwargs["n_filter_particles"] = config.n_particles
        kwargs["ess_threshold"] = config.ess_threshold_ratio
    return kwargs


def run_discrete_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> None:
    """Run discrete-time filter via cuthbert (Taylor KF, particle filter)."""

    filter_kwargs = _config_to_filter_kwargs(filter_config)
    record_kwargs = config_to_record_kwargs(filter_config)

    ys = obs_values
    t1 = int(ys.shape[0])  # this is T+1 in cuthbert's convention
    if t1 == 0:
        return

    times = obs_times

    _validate_controls(times, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)

    if ctrl_values is None:
        control_dim = dynamics.control_dim
        ctrl_values = jnp.zeros((t1, control_dim), dtype=ys.dtype)

    dt0 = times[1] - times[0]
    time_prev = jnp.concatenate([times[:1] - dt0, times[:-1]], axis=0)
    u_prev = jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0)

    cuthbert_inputs = CuthbertInputs(
        y=ys, u=ctrl_values, u_prev=u_prev, time=times, time_prev=time_prev
    )

    if isinstance(filter_config, PFConfig):
        if key is None:
            raise ValueError(
                "Particle filter requires a PRNG key: set 'crn_seed' in the filter config, "
                "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
            )
        filter_obj = _cuthbert_filter_pf(dynamics, filter_kwargs)
    elif isinstance(filter_config, (EKFConfig, EnKFConfig)):
        filter_obj = _cuthbert_filter_taylor_kf(dynamics, filter_kwargs)
    else:
        raise ValueError(
            f"Unsupported cuthbert config: {type(filter_config).__name__}. "
            "Expected EKFConfig, PFConfig, or EnKFConfig."
        )

    states = cuthbert_filter(filter_obj, cuthbert_inputs, parallel=False, key=key)
    marginal_loglik = states.log_normalizing_constant[-1]

    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)

    if isinstance(filter_config, PFConfig):
        _add_sites_pf(name, states, record_kwargs)
    else:
        _add_sites_taylor_kf(name, states, record_kwargs)


def _cuthbert_filter_pf(dynamics: DynamicalModel, filter_kwargs: dict | None = None):
    if filter_kwargs is None:
        filter_kwargs = {}

    def init_sample(key, mi: CuthbertInputs):
        return dynamics.initial_condition.sample(key)

    def propagate_sample(key, x_prev, mi: CuthbertInputs):
        dist = dynamics.state_evolution(x_prev, mi.u_prev, mi.time_prev, mi.time)  # type: ignore
        return dist.sample(key)  # type: ignore

    def log_potential(x_prev, x, mi: CuthbertInputs):
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

    def get_init_log_density(mi: CuthbertInputs):
        dist0 = dynamics.initial_condition
        state_dim = dynamics.state_dim

        def init_log_density(x):
            return jnp.asarray(dist0.log_prob(x)).sum()

        x0_lin = jnp.reshape(jnp.atleast_1d(jnp.asarray(dist0.mean)), (state_dim,))
        return init_log_density, x0_lin

    def get_dynamics_log_density(
        state: taylor.LinearizedKalmanFilterState, mi: CuthbertInputs
    ):
        def dynamics_log_density(x_prev, x):
            dist = dynamics.state_evolution(x_prev, mi.u_prev, mi.time_prev, mi.time)
            return jnp.asarray(dist.log_prob(x)).sum()

        x_prev_lin = jnp.atleast_1d(jnp.asarray(state.mean))

        dist_at_lin = dynamics.state_evolution(  # type: ignore
            x_prev_lin, mi.u_prev, mi.time_prev, mi.time
        )
        try:
            x_lin = jnp.atleast_1d(jnp.asarray(dist_at_lin.mean))  # type: ignore
        except Exception as exc:
            raise ValueError(
                "dist_at_lin.mean is not available. Linearized Kalman filter requires a mean-able distribution."
            ) from exc

        return dynamics_log_density, x_prev_lin, x_lin

    def get_observation_func(
        state: taylor.LinearizedKalmanFilterState, mi: CuthbertInputs
    ):
        def log_potential(x):
            edist = dynamics.observation_model(x, mi.u, mi.time)
            return jnp.asarray(edist.log_prob(mi.y)).sum()

        return log_potential, jnp.atleast_1d(jnp.asarray(state.mean))

    kf = taylor.build_filter(
        get_init_log_density,  # type: ignore
        get_dynamics_log_density,  # type: ignore
        get_observation_func,  # type: ignore
        associative=False,
        rtol=rtol,
        ignore_nan_dims=True,
    )
    return kf


def _add_sites_pf(
    name: str, states: particle_filter.ParticleFilterState, record_kwargs: dict
):
    log_weights = states.log_weights
    particles = states.particles
    if particles.ndim == 2:
        particles = particles[..., None]
    max_elems = record_kwargs["record_max_elems"]
    t1, n_particles, state_dim = particles.shape

    add_particles = _should_record_field(
        record_kwargs["record_filtered_particles"], particles.shape, max_elems
    )
    add_log_weights = _should_record_field(
        record_kwargs["record_filtered_log_weights"], log_weights.shape, max_elems
    )
    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], (t1, state_dim), max_elems
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], (t1, state_dim), max_elems
    )

    need_filtered_means = (
        add_mean or add_filtered_states_cov or add_filtered_states_cov_diag
    )

    if need_filtered_means:
        w = jnp.exp(log_weights)[..., None]
        filtered_means = jnp.sum(particles * w, axis=1)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        second_mom = jnp.einsum(
            "...tnj,...tnk,...tn->...tjk", particles, particles, jnp.exp(log_weights)
        )
        filtered_covariances = second_mom - jnp.einsum(
            "...tj,...tk->...tjk", filtered_means, filtered_means
        )

    if add_particles:
        numpyro.deterministic(f"{name}_filtered_particles", particles)
    if add_log_weights:
        numpyro.deterministic(f"{name}_filtered_log_weights", log_weights)
    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", filtered_means)
    if add_filtered_states_cov:
        numpyro.deterministic(f"{name}_filtered_states_cov", filtered_covariances)
    if add_filtered_states_cov_diag:
        diag_cov = jnp.diagonal(filtered_covariances, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)


def _add_sites_taylor_kf(
    name: str, states: taylor.LinearizedKalmanFilterState, record_kwargs: dict
):
    max_elems = record_kwargs["record_max_elems"]
    t1, state_dim, _ = states.chol_cov.shape

    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], states.mean.shape, max_elems
    )
    add_chol_cov = _should_record_field(
        record_kwargs["record_filtered_states_chol_cov"],
        states.chol_cov.shape,
        max_elems,
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], (t1, state_dim), max_elems
    )

    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", states.mean)
    if add_chol_cov:
        numpyro.deterministic(f"{name}_filtered_states_chol_cov", states.chol_cov)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        chol_t = jnp.transpose(states.chol_cov, (0, 2, 1))
        filtered_cov = jnp.matmul(states.chol_cov, chol_t)

    if add_filtered_states_cov:
        numpyro.deterministic(f"{name}_filtered_states_cov", filtered_cov)
    if add_filtered_states_cov_diag:
        diag_cov = jnp.diagonal(filtered_cov, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)
