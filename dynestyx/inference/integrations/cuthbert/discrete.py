import warnings
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cuthbert import filter as cuthbert_filter
from cuthbert.enkf import ensemble_kalman_filter
from cuthbert.gaussian import kalman, taylor
from cuthbert.smc import particle_filter
from cuthbertlib.resampling import (
    adaptive,
    multinomial,
    stop_gradient_decorator,
    systematic,
)
from numpyro.distributions import Distribution

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    EKFConfig,
    EnKFConfig,
    KFConfig,
    PFConfig,
    _config_to_record_kwargs,
)
from dynestyx.inference.integrations.utils import (
    covariance_from_cholesky,
    particles_to_delta_mixtures,
)
from dynestyx.models import (
    DynamicalModel,
    GaussianObservation,
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
    is_first_step: jax.Array  # (T+1,) bool — True only at index 1


def _extract_gaussian_chol(d: dist.Distribution, obs_dim: int) -> jax.Array:
    """Extract a Cholesky factor of the covariance from a Gaussian distribution."""
    if isinstance(d, dist.MultivariateNormal):
        return jnp.asarray(d.scale_tril)
    if isinstance(d, dist.Independent) and isinstance(d.base_dist, dist.Normal):
        scale = jnp.atleast_1d(jnp.asarray(d.base_dist.scale))
    elif isinstance(d, dist.Normal):
        scale = jnp.atleast_1d(jnp.asarray(d.scale))
    else:
        raise TypeError(
            "cuthbert EnKF requires Gaussian observation distributions. "
            "Expected LinearGaussianObservation, GaussianObservation, or a "
            "callable returning Normal, Independent(Normal), or "
            f"MultivariateNormal; got {type(d).__name__}."
        )
    if scale.size == 1 and obs_dim > 1:
        scale = jnp.full((obs_dim,), scale[0])
    return jnp.diag(scale)


def _check_state_independent_noise(
    chol_R_at_x0: jax.Array, probe_dist_at_x1: dist.Distribution, obs_dim: int
) -> None:
    """Raise if the observation noise covariance varies with state.

    cuthbert's EnKF API resolves ``chol_R`` once per step (before the ensemble
    update), so it may depend on time/controls but NOT on the latent state.
    Silently using a state-dependent scale would freeze it at the probe state
    and misrepresent the noise. Detect this under concrete evaluation and raise.
    Under JAX tracing, the comparison yields a tracer and we skip the check —
    the constraint is documented in the filter docstring.
    """
    chol_R_at_x1 = _extract_gaussian_chol(probe_dist_at_x1, obs_dim)
    try:
        equal = bool(
            jnp.asarray(chol_R_at_x0).shape == jnp.asarray(chol_R_at_x1).shape
            and jnp.allclose(chol_R_at_x0, chol_R_at_x1)
        )
    except jax.errors.TracerBoolConversionError:
        return
    if not equal:
        raise ValueError(
            "cuthbert EnKF requires state-independent observation noise, but "
            "the observation scale changes with the latent state (heteroscedastic "
            "noise). The EnKF API resolves chol_R once per step before the "
            "ensemble update, so a state-dependent scale cannot be honoured. "
            "Either make the noise depend only on time/controls, or use a "
            "particle filter (PFConfig)."
        )


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
    elif isinstance(config, EnKFConfig):
        kwargs["n_particles"] = config.n_particles
        kwargs["inflation"] = (
            config.inflation_delta if config.inflation_delta is not None else 0.0
        )
        if config.perturb_measurements is not None:
            kwargs["perturbed_obs"] = config.perturb_measurements
    return kwargs


def _drop_cuthbert_dummy_step(states, *, obs_len: int):
    """Drop cuthbert's leading dummy state from every time-indexed leaf."""
    raw_len = obs_len + 1

    def _drop_if_time_leaf(leaf):
        shape = getattr(leaf, "shape", None)
        ndim = getattr(leaf, "ndim", None)
        if ndim is None and shape is not None:
            ndim = len(shape)
        if shape is not None and ndim is not None and ndim > 0 and shape[0] == raw_len:
            return leaf[1:]
        return leaf

    return jax.tree.map(_drop_if_time_leaf, states)


def compute_cuthbert_filter(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX cuthbert filter computation (no numpyro side-effects).

    Returns:
        tuple: (marginal_loglik, states) where states are aligned to obs_times.
    """
    filter_kwargs = _config_to_filter_kwargs(filter_config)

    ys = obs_values
    obs_len = int(ys.shape[0])
    times = obs_times

    if ctrl_values is None:
        control_dim = dynamics.control_dim
        ctrl_values = jnp.zeros((obs_len, control_dim), dtype=ys.dtype)
    elif ctrl_values.shape[0] > obs_len:
        inds = jnp.searchsorted(ctrl_times, times, side="left")
        ctrl_values = ctrl_values[inds]

    dt0 = times[1] - times[0]
    time_prev = jnp.concatenate([times[:1] - dt0, times[:-1]], axis=0)
    u_prev = jnp.concatenate([ctrl_values[:1], ctrl_values[:-1]], axis=0)

    dummy_y = jnp.zeros_like(ys[:1])
    dummy_u = jnp.zeros_like(ctrl_values[:1])
    dummy_time = jnp.zeros_like(times[:1])

    cuthbert_inputs = CuthbertInputs(
        y=jnp.concatenate([dummy_y, ys], axis=0),
        u=jnp.concatenate([dummy_u, ctrl_values], axis=0),
        u_prev=jnp.concatenate([dummy_u, u_prev], axis=0),
        time=jnp.concatenate([dummy_time, times], axis=0),
        time_prev=jnp.concatenate([dummy_time, time_prev], axis=0),
        is_first_step=jnp.arange(obs_len + 1) == 1,
    )

    if isinstance(filter_config, PFConfig):
        if key is None:
            raise ValueError(
                "Particle filter requires a PRNG key: set 'crn_seed' in the filter config, "
                "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
            )
        filter_obj = _cuthbert_filter_pf(dynamics, filter_kwargs)
    elif isinstance(filter_config, EnKFConfig):
        if key is None:
            raise ValueError(
                "Ensemble Kalman filter requires a PRNG key: set 'crn_seed' in the filter config, "
                "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
            )
        filter_obj = _cuthbert_filter_enkf(dynamics, filter_kwargs)
    elif isinstance(filter_config, KFConfig):
        filter_obj = _cuthbert_filter_kalman(dynamics, filter_kwargs)
    elif isinstance(filter_config, EKFConfig):
        filter_obj = _cuthbert_filter_taylor_kf(dynamics, filter_kwargs)
    else:
        raise ValueError(
            f"Unsupported cuthbert config: {type(filter_config).__name__}. "
            "Expected KFConfig, EKFConfig, EnKFConfig, PFConfig."
        )

    raw_states = cuthbert_filter(filter_obj, cuthbert_inputs, parallel=False, key=key)
    marginal_loglik = raw_states.log_normalizing_constant[-1]
    states = _drop_cuthbert_dummy_step(raw_states, obs_len=obs_len)
    return marginal_loglik, states


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
    """Run discrete-time filter via cuthbert (Kalman, Taylor KF, EnKF, PF).

    Returns:
        list[dist.Distribution]: Filtered state distributions at each obs time.
    """
    obs_len = int(obs_values.shape[0])
    if obs_len == 0:
        return []

    marginal_loglik, states = compute_cuthbert_filter(
        dynamics,
        filter_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    record_kwargs = _config_to_record_kwargs(filter_config)

    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)

    if isinstance(filter_config, PFConfig):
        _add_sites_pf(name, states, record_kwargs)
        particles = states.particles
        if particles.ndim == 2:
            particles = particles[..., None]
        return particles_to_delta_mixtures(particles, states.log_weights)
    else:
        _add_sites_gaussian_filter(name, states, record_kwargs)
        mean = states.mean
        chol_cov = states.chol_cov
        cov = covariance_from_cholesky(chol_cov)
        return [
            dist.MultivariateNormal(mean[i], covariance_matrix=cov[i])
            for i in range(mean.shape[0])
        ]


def _cuthbert_filter_pf(dynamics: DynamicalModel, filter_kwargs: dict | None = None):
    if filter_kwargs is None:
        filter_kwargs = {}

    def init_sample(key, mi: CuthbertInputs):
        return dynamics.initial_condition.sample(key)

    def propagate_sample(key, x_prev, mi: CuthbertInputs):
        def _noop(key, x_prev, mi):
            return x_prev

        def _evolve(key, x_prev, mi):
            d = dynamics.state_evolution(x_prev, mi.u_prev, mi.time_prev, mi.time)  # type: ignore
            return d.sample(key)  # type: ignore

        return jax.lax.cond(mi.is_first_step, _noop, _evolve, key, x_prev, mi)

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
    )
    return pf


def _cuthbert_filter_enkf(dynamics: DynamicalModel, filter_kwargs: dict | None = None):
    if filter_kwargs is None:
        filter_kwargs = {}

    state_dim = dynamics.state_dim
    obs_dim = dynamics.observation_dim

    obs_model = dynamics.observation_model
    if not isinstance(obs_model, (LinearGaussianObservation, GaussianObservation)):
        probe_u = jnp.zeros(())
        probe_t = jnp.zeros(())
        try:
            probe_d0: Distribution | None = obs_model(
                jnp.zeros((state_dim,)), probe_u, probe_t
            )
            probe_d1: Distribution | None = obs_model(
                jnp.ones((state_dim,)), probe_u, probe_t
            )
        except Exception:
            warnings.warn(
                "Failed to probe observation model for state-independent noise check. Please ensure the observation model is state-independent."
            )
        if probe_d0 is not None and probe_d1 is not None:
            chol0 = _extract_gaussian_chol(probe_d0, obs_dim)
            _check_state_independent_noise(chol0, probe_d1, obs_dim)

    def init_sample(key, mi: CuthbertInputs):
        return jnp.atleast_1d(jnp.asarray(dynamics.initial_condition.sample(key)))

    def get_dynamics(mi: CuthbertInputs):
        def dynamics_fn(x, key):
            def _noop(key):
                return x

            def _evolve(key):
                d = dynamics.state_evolution(x, mi.u_prev, mi.time_prev, mi.time)  # type: ignore
                return jnp.atleast_1d(jnp.asarray(d.sample(key)))  # type: ignore

            return jax.lax.cond(mi.is_first_step, _noop, _evolve, key)

        return dynamics_fn

    def get_observations(mi: CuthbertInputs):
        obs_model = dynamics.observation_model
        y = jnp.atleast_1d(jnp.asarray(mi.y))

        if isinstance(obs_model, LinearGaussianObservation):
            H = jnp.asarray(obs_model.H)
            chol_R = jnp.linalg.cholesky(jnp.atleast_2d(jnp.asarray(obs_model.R)))
            bias = (
                jnp.zeros((obs_dim,), dtype=y.dtype)
                if obs_model.bias is None
                else jnp.atleast_1d(jnp.asarray(obs_model.bias))
            )
            D = None if obs_model.D is None else jnp.asarray(obs_model.D)

            def observation_fn(x):
                loc = H @ x + bias
                if D is not None:
                    loc = loc + D @ jnp.atleast_1d(jnp.asarray(mi.u))
                return jnp.atleast_1d(jnp.asarray(loc))

            return observation_fn, chol_R, y
        elif isinstance(obs_model, GaussianObservation):
            chol_R = jnp.linalg.cholesky(jnp.atleast_2d(jnp.asarray(obs_model.R)))

            def observation_fn(x):
                return jnp.atleast_1d(jnp.asarray(obs_model.h(x, mi.u, mi.time)))

            return observation_fn, chol_R, y
        else:
            probe_x0 = jnp.zeros((state_dim,), dtype=y.dtype)
            probe_x1 = jnp.ones((state_dim,), dtype=y.dtype)
            probe_dist = obs_model(probe_x0, mi.u, mi.time)
            chol_R = _extract_gaussian_chol(probe_dist, obs_dim)
            _check_state_independent_noise(
                chol_R, obs_model(probe_x1, mi.u, mi.time), obs_dim
            )

            def observation_fn(x):
                edist = obs_model(x, mi.u, mi.time)
                if not (
                    isinstance(edist, (dist.MultivariateNormal, dist.Normal))
                    or (
                        isinstance(edist, dist.Independent)
                        and isinstance(edist.base_dist, dist.Normal)
                    )
                ):
                    raise TypeError(
                        "cuthbert EnKF observation callable must keep returning "
                        "Gaussian distributions; got "
                        f"{type(edist).__name__}."
                    )
                return jnp.atleast_1d(jnp.asarray(edist.mean))

            return observation_fn, chol_R, y

    return ensemble_kalman_filter.build_filter(
        init_sample=init_sample,  # type: ignore
        get_dynamics=get_dynamics,  # type: ignore
        get_observations=get_observations,  # type: ignore
        n_particles=int(filter_kwargs.get("n_particles", 30)),
        inflation=float(filter_kwargs.get("inflation", 0.0)),
        perturbed_obs=bool(filter_kwargs.get("perturbed_obs", True)),
    )


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
        def _noop(mi):
            return (
                jnp.eye(state_dim, dtype=m0.dtype),
                jnp.zeros((state_dim,), dtype=m0.dtype),
                jnp.zeros((state_dim, state_dim), dtype=m0.dtype),
            )

        def _evolve(mi):
            c = evo_bias
            if B is not None:
                c = c + B @ jnp.atleast_1d(jnp.asarray(mi.u_prev))
            return A, c, chol_Q

        return jax.lax.cond(mi.is_first_step, _noop, _evolve, mi)

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
            normal_logp = jnp.asarray(
                dynamics.state_evolution(
                    x_prev, mi.u_prev, mi.time_prev, mi.time
                ).log_prob(x)
            ).sum()
            # Identity dynamics with near-zero noise for the noop first step.
            noop_logp = -1e10 * jnp.sum((x - x_prev) ** 2)
            return jnp.where(mi.is_first_step, noop_logp, normal_logp)

        x_prev_lin = jnp.atleast_1d(jnp.asarray(state.mean))

        dist_at_lin = dynamics.state_evolution(  # type: ignore
            x_prev_lin, mi.u_prev, mi.time_prev, mi.time
        )
        try:
            x_lin = jnp.atleast_1d(jnp.asarray(dist_at_lin.mean))  # type: ignore
        except (AttributeError, NotImplementedError) as exc:
            raise ValueError(
                "dist_at_lin.mean is not available. Linearized Kalman filter requires a mean-able distribution."
            ) from exc

        # On the first step, use identity linearization (x_lin = x_prev_lin).
        x_lin = jnp.where(mi.is_first_step, x_prev_lin, x_lin)

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
    t_len, n_particles, state_dim = particles.shape

    add_particles = _should_record_field(
        record_kwargs["record_filtered_particles"], particles.shape, max_elems
    )
    add_log_weights = _should_record_field(
        record_kwargs["record_filtered_log_weights"], log_weights.shape, max_elems
    )
    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], (t_len, state_dim), max_elems
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"],
        (t_len, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], (t_len, state_dim), max_elems
    )

    need_filtered_means = (
        add_mean or add_filtered_states_cov or add_filtered_states_cov_diag
    )

    if need_filtered_means:
        w = jax.nn.softmax(log_weights, axis=1)[..., None]  # (T, n_particles, 1)
        filtered_means = jnp.sum(particles * w, axis=1)  # (T, state_dim)

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


def _add_sites_gaussian_filter(
    name: str,
    states: taylor.LinearizedKalmanFilterState | ensemble_kalman_filter.EnKFState,
    record_kwargs: dict,
):
    max_elems = record_kwargs["record_max_elems"]
    mean = states.mean
    chol_cov = states.chol_cov
    t_len, state_dim, _ = chol_cov.shape

    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], mean.shape, max_elems
    )
    add_chol_cov = _should_record_field(
        record_kwargs["record_filtered_states_chol_cov"],
        chol_cov.shape,
        max_elems,
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"],
        (t_len, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], (t_len, state_dim), max_elems
    )

    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", mean)
    if add_chol_cov:
        numpyro.deterministic(f"{name}_filtered_states_chol_cov", chol_cov)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        filtered_cov = covariance_from_cholesky(chol_cov)

    if add_filtered_states_cov:
        numpyro.deterministic(f"{name}_filtered_states_cov", filtered_cov)
    if add_filtered_states_cov_diag:
        diag_cov = jnp.diagonal(filtered_cov, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)
