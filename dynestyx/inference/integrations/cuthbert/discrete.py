from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cuthbert import filter as cuthbert_filter
from cuthbert.gaussian import kalman, taylor
from cuthbert.smc import particle_filter
from cuthbertlib.resampling import (
    adaptive,
    multinomial,
    stop_gradient_decorator,
    systematic,
)

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    EKFConfig,
    KFConfig,
    PFConfig,
    _config_to_record_kwargs,
)
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from dynestyx.utils import _should_record_field


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
        kwargs["resampling_base_method"] = config.resampling_method.base_method
        kwargs["resampling_differential_method"] = (
            config.resampling_method.differential_method
        )
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
) -> list[dist.Distribution]:
    """Run discrete-time filter via cuthbert (Kalman, Taylor KF, particle filter).

    Returns:
        list[dist.Distribution]: Filtered state distributions at each obs time.
    """
    filter_kwargs = _config_to_filter_kwargs(filter_config)
    record_kwargs = _config_to_record_kwargs(filter_config)

    ys = obs_values
    t1 = int(ys.shape[0])  # this is T+1 in cuthbert's convention
    if t1 == 0:
        return []

    times = obs_times

    if ctrl_values is None:
        control_dim = dynamics.control_dim
        ctrl_values = jnp.zeros((t1, control_dim), dtype=ys.dtype)
    elif ctrl_values.shape[0] > t1:
        # ctrl spans union of obs_times and predict_times; filter needs ctrl at obs_times only
        inds = jnp.searchsorted(ctrl_times, times, side="left")
        ctrl_values = ctrl_values[inds]

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
    elif isinstance(filter_config, KFConfig):
        filter_obj = _cuthbert_filter_kalman(dynamics, filter_kwargs)
    elif isinstance(filter_config, EKFConfig):
        filter_obj = _cuthbert_filter_taylor_kf(dynamics, filter_kwargs)
    else:
        raise ValueError(
            f"Unsupported cuthbert config: {type(filter_config).__name__}. "
            "Expected KFConfig, EKFConfig, PFConfig."
        )

    states = cuthbert_filter(filter_obj, cuthbert_inputs, parallel=False, key=key)
    marginal_loglik = states.log_normalizing_constant[-1]

    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)

    if isinstance(filter_config, PFConfig):
        _add_sites_pf(name, states, record_kwargs)
        # PF: mixture of deltas
        particles = states.particles
        if particles.ndim == 2:
            particles = particles[..., None]
        log_weights = states.log_weights
        log_weights_norm = log_weights - jax.scipy.special.logsumexp(
            log_weights, axis=1, keepdims=True
        )
        result = []
        for i in range(particles.shape[0]):
            mixing = dist.Categorical(logits=log_weights_norm[i])
            comps = dist.Delta(particles[i], event_dim=1)
            result.append(dist.MixtureSameFamily(mixing, comps))
        return result
    else:
        _add_sites_taylor_kf(name, states, record_kwargs)
        # KF/EKF: states.mean (T+1, state_dim), states.chol_cov (T+1, state_dim, state_dim)
        chol_t = jnp.transpose(states.chol_cov, (0, 2, 1))
        cov = jnp.matmul(states.chol_cov, chol_t)
        return [
            dist.MultivariateNormal(states.mean[i], covariance_matrix=cov[i])
            for i in range(states.mean.shape[0])
        ]


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
    base_method = filter_kwargs.get("resampling_base_method", "systematic")
    if base_method == "systematic":
        base_resampling_fn = systematic.resampling
    elif base_method == "multinomial":
        base_resampling_fn = multinomial.resampling
    else:
        raise ValueError(
            f"Unsupported cuthbert PF base resampling method: {base_method!r}. "
            "Expected one of: 'systematic', 'multinomial'."
        )

    differential_method = filter_kwargs.get(
        "resampling_differential_method", "stop_gradient"
    )
    if differential_method == "stop_gradient":
        base_resampling_fn = stop_gradient_decorator(base_resampling_fn)
    elif differential_method == "straight_through":
        pass
    else:
        raise ValueError(
            "Unsupported cuthbert PF differential resampling method: "
            f"{differential_method!r}. Expected one of: "
            "'stop_gradient', 'straight_through'."
        )

    resampling_fn = adaptive.ess_decorator(base_resampling_fn, ess_threshold)

    pf = particle_filter.build_filter(
        init_sample=init_sample,  # type: ignore
        propagate_sample=propagate_sample,  # type: ignore
        log_potential=log_potential,  # type: ignore
        n_filter_particles=int(filter_kwargs.get("n_filter_particles", 1_000)),
        resampling_fn=resampling_fn,  # type: ignore
        consume_first_observation=True,
    )
    return pf


def _cuthbert_filter_kalman(
    dynamics: DynamicalModel, filter_kwargs: dict | None = None
):
    if filter_kwargs is None:
        filter_kwargs = {}

    if not (
        isinstance(dynamics.state_evolution, LinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, LinearGaussianObservation)
        and isinstance(dynamics.initial_condition, dist.MultivariateNormal)
    ):
        raise TypeError(
            "cuthbert Kalman filter expects a DynamicalModel with "
            "LinearGaussianStateEvolution and LinearGaussianObservation, and "
            "initial_condition as MultivariateNormal."
        )

    evo = dynamics.state_evolution
    obs = dynamics.observation_model
    ic = dynamics.initial_condition

    state_dim = dynamics.state_dim
    obs_dim = dynamics.observation_dim

    m0 = jnp.reshape(jnp.atleast_1d(jnp.asarray(ic.loc)), (state_dim,))
    chol_P0 = jnp.linalg.cholesky(jnp.asarray(ic.covariance_matrix))

    A = jnp.asarray(evo.A)
    chol_Q = jnp.linalg.cholesky(jnp.asarray(evo.cov))

    H = jnp.asarray(obs.H)
    chol_R = jnp.linalg.cholesky(jnp.asarray(obs.R))

    evo_bias = (
        jnp.zeros((state_dim,), dtype=m0.dtype)
        if evo.bias is None
        else jnp.reshape(jnp.atleast_1d(jnp.asarray(evo.bias)), (state_dim,))
    )
    obs_bias = (
        jnp.zeros((obs_dim,), dtype=m0.dtype)
        if obs.bias is None
        else jnp.reshape(jnp.atleast_1d(jnp.asarray(obs.bias)), (obs_dim,))
    )

    B = None if evo.B is None else jnp.asarray(evo.B)
    D = None if obs.D is None else jnp.asarray(obs.D)

    def get_init_params(mi: CuthbertInputs):
        return m0, chol_P0

    def get_dynamics_params(mi: CuthbertInputs):
        c = evo_bias
        if B is not None:
            c = c + B @ jnp.atleast_1d(jnp.asarray(mi.u_prev))

        return A, c, chol_Q

    def get_observation_params(mi: CuthbertInputs):
        d = obs_bias
        if D is not None:
            d = d + D @ jnp.atleast_1d(jnp.asarray(mi.u))
        y = jnp.atleast_1d(jnp.asarray(mi.y))
        return H, d, chol_R, y

    return kalman.build_filter(
        get_init_params,  # type: ignore
        get_dynamics_params,  # type: ignore
        get_observation_params,  # type: ignore
        consume_first_observation=True,
    )


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
        consume_first_observation=True,
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
        log_weights_norm = log_weights - jax.scipy.special.logsumexp(
            log_weights, axis=1, keepdims=True
        )
        w = jnp.exp(log_weights_norm)[..., None]  # (T+1, n_particles, 1)
        filtered_means = jnp.sum(particles * w, axis=1)  # (T+1, state_dim)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        second_mom = jnp.einsum(
            "...tnj,...tnk,...tn->...tjk", particles, particles, w.squeeze(-1)
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
