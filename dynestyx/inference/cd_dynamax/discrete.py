"""Discrete-time filters via cd-dynamax (dynamax): KF, EKF, UKF."""

from collections.abc import Callable

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
from dynestyx.dynamical_models import (
    Context,
    DynamicalModel,
    LinearGaussianStateEvolution,
)
from dynestyx.inference.cd_dynamax.utils import gaussian_to_nlgssm_params
from dynestyx.observations import LinearGaussianObservation
from dynestyx.utils import (
    _get_controls,
    _should_record_field,
    _validate_control_dim,
)

_CD_DYNAMAX_DISCRETE_FILTER_TYPES: list[str] = ["kf", "ekf", "ukf"]


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
            x0_mean=jnp.asarray(ic.loc),
            x0_cov=jnp.asarray(ic.covariance_matrix),
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
    emissions = obs_traj.values
    times = jnp.asarray(obs_traj.times)
    T1 = emissions.shape[0]
    _, ctrl_values = _get_controls(context, times)
    _validate_control_dim(dynamics, ctrl_values)
    control_dim = dynamics.control_dim
    inputs = ctrl_values if ctrl_values is not None else jnp.zeros((T1, control_dim))

    params_nl = gaussian_to_nlgssm_params(dynamics)
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


def run_discrete_filter(
    name: str,
    filter_type: str,
    dynamics: DynamicalModel,
    context: Context,
    key: jax.Array | None = None,
    filter_kwargs: dict | None = None,
    record_kwargs: dict | None = None,
) -> None:
    """Run discrete-time filter via cd-dynamax (KF, EKF, UKF)."""
    if filter_kwargs is None:
        filter_kwargs = {}
    if record_kwargs is None:
        record_kwargs = {}

    ft = filter_type.lower()
    if ft == "kf":
        _filter_discrete_time_dynamax_kf(name, dynamics, context, record_kwargs)
    elif ft == "ekf":
        num_iter = filter_kwargs.get("num_iter", 1)
        _filter_discrete_time_dynamax_ekf(
            name, dynamics, context, record_kwargs, num_iter=num_iter
        )
    elif ft == "ukf":
        hyperparams = filter_kwargs.get("ukf_hyperparams")
        _filter_discrete_time_dynamax_ukf(
            name, dynamics, context, record_kwargs, hyperparams=hyperparams
        )
    else:
        raise ValueError(
            f"Unsupported cd-dynamax discrete filter type: {filter_type}. "
            f"Expected one of {_CD_DYNAMAX_DISCRETE_FILTER_TYPES}."
        )
