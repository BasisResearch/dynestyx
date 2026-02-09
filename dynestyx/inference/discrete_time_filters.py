from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.builders import build_params
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    PosteriorGSSMFiltered,
    lgssm_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    unscented_kalman_filter,
)
from cuthbert import filter as cuthbert_filter
from cuthbert.gaussian import taylor
from cuthbert.smc import particle_filter

from dynestyx.cuthbert_patches import systematic_resampling
from dynestyx.dynamical_models import (
    Context,
    DynamicalModel,
    LinearGaussianStateEvolution,
)
from dynestyx.inference.cd_dynamax.utils import lti_to_nlgssm_params
from dynestyx.observations import LinearGaussianObservation
from dynestyx.utils import (
    _get_controls,
    _should_record_field,
    _validate_control_dim,
)

_DISCRETE_FILTER_TYPES: list[str] = ["default", "taylor_kf", "pf", "kf", "ekf", "ukf"]


class _CuthbertInputs(NamedTuple):
    """Model inputs pytree for cuthbert; leading time dim must be T+1."""

    y: jax.Array  # (T+1, emission_dim)
    u: jax.Array  # (T+1, control_dim) or (T+1, 0)
    u_prev: jax.Array  # (T+1, control_dim) or (T+1, 0)
    time: jax.Array  # (T+1,)
    time_prev: jax.Array  # (T+1,)


def _lti_to_lgssm_params(dynamics: DynamicalModel):
    """Build dynamax ParamsLGSSM via builders.build_params from an LTI model.

    Supports either:
    - An LTI_discretetime (or any model with .F, .Q, .H, .R, .initial_mean, .initial_cov, [.B, .b, .D, .d]), or
    - A DynamicalModel with LinearGaussianStateEvolution and LinearGaussianObservation (A, Q, B, b -> F,Q,B,b; H, R, D, d).
    """
    state_dim = dynamics.state_dim
    emission_dim = dynamics.observation_dim
    control_dim = dynamics.control_dim

    if (
        isinstance(dynamics.state_evolution, LinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, LinearGaussianObservation)
        and isinstance(dynamics.initial_condition, dist.MultivariateNormal)
    ):
        evo = dynamics.state_evolution
        obs = dynamics.observation_model
        ic = dynamics.initial_condition
        return build_params(
            state_dim=state_dim,
            emission_dim=emission_dim,
            input_dim=control_dim,
            dynamics_weights=evo.A,
            has_dynamics_bias=evo.bias is not None,
            dynamics_bias=evo.bias,
            dynamics_input_weights=evo.B,
            dynamics_cov=evo.cov,
            emission_weights=obs.H,
            emission_input_weights=obs.D,
            has_emissions_bias=obs.bias is not None,
            emission_bias=obs.bias,
            emission_cov=obs.R,
            x0_mean=ic.loc,
            x0_cov=ic.covariance_matrix,
        )
    else:
        raise TypeError(
            "filter_type='kf' expects a DynamicalModel with LinearGaussianStateEvolution and LinearGaussianObservation and initial_condition as MultivariateNormal."
        )


def _filter_discrete_time_dynamax_kf(
    name: str,
    dynamics: DynamicalModel,
    context: Context,
    record_kwargs: dict,
) -> None:
    """Run dynamax Kalman filter for LTI_discretetime and add factor + sites."""
    obs_traj = context.observations
    if obs_traj.values is None or obs_traj.times is None:
        return
    if isinstance(obs_traj.values, dict):
        raise ValueError("obs_traj.values must be an Array, not a dict")
    emissions = obs_traj.values
    times = jnp.asarray(obs_traj.times)
    T1 = emissions.shape[0]
    _, ctrl_values = _get_controls(context, times)
    _validate_control_dim(dynamics, ctrl_values)
    control_dim = dynamics.control_dim
    if ctrl_values is not None:
        inputs = ctrl_values
    else:
        inputs = jnp.zeros((T1, control_dim))

    params = _lti_to_lgssm_params(dynamics)
    posterior = lgssm_filter(params, emissions, inputs=inputs)

    marginal_loglik = posterior.marginal_loglik
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)
    _add_kf_sites(name, posterior, record_kwargs)


def _run_nlgssm_filter(
    name: str,
    dynamics: DynamicalModel,
    context: Context,
    record_kwargs: dict,
    run_filter: Callable,
) -> None:
    """Common setup for EKF/UKF: get emissions/inputs, run filter, add factor + sites."""
    obs_traj = context.observations
    if obs_traj.values is None or obs_traj.times is None:
        return
    if isinstance(obs_traj.values, dict):
        raise ValueError("obs_traj.values must be an Array, not a dict")
    emissions = obs_traj.values
    times = jnp.asarray(obs_traj.times)
    T1 = emissions.shape[0]
    _, ctrl_values = _get_controls(context, times)
    _validate_control_dim(dynamics, ctrl_values)
    control_dim = dynamics.control_dim
    inputs = ctrl_values if ctrl_values is not None else jnp.zeros((T1, control_dim))

    params_nl = lti_to_nlgssm_params(dynamics)
    posterior = run_filter(params_nl, emissions, inputs)

    marginal_loglik = posterior.marginal_loglik
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)
    _add_kf_sites(name, posterior, record_kwargs)


def _filter_discrete_time_dynamax_ekf(
    name: str,
    dynamics: DynamicalModel,
    context: Context,
    record_kwargs: dict,
    num_iter: int = 1,
) -> None:
    """Run dynamax EKF for LTI and add factor + sites."""

    def run_filter(params_nl, emissions, inputs):
        return extended_kalman_filter(
            params_nl, emissions, num_iter=num_iter, inputs=inputs
        )

    _run_nlgssm_filter(name, dynamics, context, record_kwargs, run_filter)


def _filter_discrete_time_dynamax_ukf(
    name: str,
    dynamics: DynamicalModel,
    context: Context,
    record_kwargs: dict,
    hyperparams: UKFHyperParams | None = None,
) -> None:
    """Run dynamax UKF for LTI and add factor + sites."""
    if hyperparams is None:
        hyperparams = UKFHyperParams()

    def run_filter(params_nl, emissions, inputs):
        return unscented_kalman_filter(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )

    _run_nlgssm_filter(name, dynamics, context, record_kwargs, run_filter)


def _add_kf_sites(
    name: str, posterior: PosteriorGSSMFiltered, record_kwargs: dict
) -> None:
    """Add filtered means/covariances as deterministic sites (dynamax KF posterior)."""
    max_elems = record_kwargs.get("record_max_elems", 100_000)
    if posterior.filtered_means is None:
        return
    means = posterior.filtered_means
    covs = posterior.filtered_covariances
    T1, state_dim = means.shape
    add_mean = _should_record_field(
        record_kwargs.get("record_filtered_states_mean"), means.shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs.get("record_filtered_states_cov"),
        (T1, state_dim, state_dim),
        max_elems,
    )
    add_cov_diag = _should_record_field(
        record_kwargs.get("record_filtered_states_cov_diag"), (T1, state_dim), max_elems
    )
    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", means)
    if add_cov and covs is not None:
        numpyro.deterministic(f"{name}_filtered_states_cov", covs)
    if add_cov_diag and covs is not None:
        diag_cov = jnp.diagonal(covs, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)


def _filter_discrete_time(
    name: str,
    filter_type: str,
    dynamics: DynamicalModel,
    context: Context,
    key: jax.Array | None = None,
    filter_kwargs: dict | None = None,
    record_kwargs: dict = {},
):
    """Discrete-time marginal likelihood via cuthbert or dynamax (kf).

    Args:
        name: Name of the factor.
        filter_type: Type of filter to use (taylor_kf, pf, kf).
        dynamics: Dynamical model to filter.
        context: Context containing the observations and controls.
        key: Random key for the filter.
        filter_kwargs: Keyword arguments for the filter.
        record_kwargs: Keyword arguments for recording the filtered states and their covariances.
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    if filter_type.lower() == "kf":
        _filter_discrete_time_dynamax_kf(name, dynamics, context, record_kwargs)
        return
    if filter_type.lower() == "ekf":
        num_iter = filter_kwargs.get("num_iter", 1)
        _filter_discrete_time_dynamax_ekf(
            name, dynamics, context, record_kwargs, num_iter=num_iter
        )
        return
    if filter_type.lower() == "ukf":
        hyperparams = filter_kwargs.get("ukf_hyperparams")
        _filter_discrete_time_dynamax_ukf(
            name, dynamics, context, record_kwargs, hyperparams=hyperparams
        )
        return

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
        control_dim = dynamics.control_dim
        ctrl_values = jnp.zeros((T1, control_dim), dtype=ys.dtype)

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

    # Add the marginal log likelihood as a numpyro factor
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)

    # Add the marginal log likelihood as a deterministic site for easy access.
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)

    # Optionally record the filtered states and their covariances as deterministic sites for easy access.
    if filter_type.lower() in ["taylor_kf", "default"]:
        _add_sites_taylor_kf(name, states, record_kwargs)
    elif filter_type.lower() == "pf":
        _add_sites_pf(name, states, record_kwargs)
    else:
        raise ValueError(
            f"Invalid filter type: {filter_type}. Valid types: {_DISCRETE_FILTER_TYPES}"
        )


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
        state_dim = dynamics.state_dim

        def init_log_density(x):
            return jnp.asarray(dist0.log_prob(x)).sum()

        # Ensure (state_dim,) so cuthbert's linearize_log_density gets 2D Hessian; scalar mean would give 0-dim and jnp.diag fails.
        x0_lin = jnp.reshape(jnp.atleast_1d(jnp.asarray(dist0.mean)), (state_dim,))
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


def _add_sites_pf(
    name: str, states: particle_filter.ParticleFilterState, record_kwargs: dict
):
    # Compute filtered means and covariances from the particles using the weights.
    # particles (T+1, n_particles, state_dim) or (T+1, n_particles) when state_dim=1
    log_weights = states.log_weights
    particles = states.particles
    if particles.ndim == 2:
        particles = jnp.expand_dims(particles, axis=-1)  # (T+1, n_particles, 1)
    max_elems = record_kwargs["record_max_elems"]
    T1, n_particles, state_dim = particles.shape

    add_particles = _should_record_field(
        record_kwargs.get("record_filtered_particles"), particles.shape, max_elems
    )
    add_log_weights = _should_record_field(
        record_kwargs.get("record_filtered_log_weights"), log_weights.shape, max_elems
    )
    add_mean = _should_record_field(
        record_kwargs.get("record_filtered_states_mean"), (T1, state_dim), max_elems
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs.get("record_filtered_states_cov"),
        (T1, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs.get("record_filtered_states_cov_diag"), (T1, state_dim), max_elems
    )

    need_filtered_means = (
        add_mean or add_filtered_states_cov or add_filtered_states_cov_diag
    )

    if need_filtered_means:
        w = jnp.exp(log_weights)[..., None]  # (T+1, n_particles, 1) for broadcasting
        filtered_means = jnp.sum(particles * w, axis=1)  # (T+1, state_dim)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        # Weighted covariance: E[xx'] - E[x]E[x]'
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
    max_elems = record_kwargs.get("record_max_elems", 100_000)
    T1, state_dim, _ = states.chol_cov.shape

    add_mean = _should_record_field(
        record_kwargs.get("record_filtered_states_mean"), states.mean.shape, max_elems
    )
    add_chol_cov = _should_record_field(
        record_kwargs.get("record_filtered_states_chol_cov"),
        states.chol_cov.shape,
        max_elems,
    )
    add_filtered_states_cov = _should_record_field(
        record_kwargs.get("record_filtered_states_cov"),
        (T1, state_dim, state_dim),
        max_elems,
    )
    add_filtered_states_cov_diag = _should_record_field(
        record_kwargs.get("record_filtered_states_cov_diag"), (T1, state_dim), max_elems
    )

    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", states.mean)
    if add_chol_cov:
        numpyro.deterministic(f"{name}_filtered_states_chol_cov", states.chol_cov)

    if add_filtered_states_cov or add_filtered_states_cov_diag:
        chol_T = jnp.transpose(states.chol_cov, (0, 2, 1))
        filtered_cov = jnp.matmul(states.chol_cov, chol_T)

    if add_filtered_states_cov:
        numpyro.deterministic(f"{name}_filtered_states_cov", filtered_cov)
    if add_filtered_states_cov_diag:
        diag_cov = jnp.diagonal(filtered_cov, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)
