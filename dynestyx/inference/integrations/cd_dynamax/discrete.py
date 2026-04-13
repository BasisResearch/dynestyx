"""Discrete-time filters via cd-dynamax (dynamax): KF, EKF, UKF."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    PosteriorGSSMFiltered,
    lgssm_filter,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    unscented_kalman_filter,
)

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    EKFConfig,
    KFConfig,
    UKFConfig,
    _config_to_record_kwargs,
)
from dynestyx.inference.integrations.cd_dynamax.utils import gaussian_to_nlgssm_params
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from dynestyx.utils import _should_record_field


def _lti_to_lgssm_params(dynamics: DynamicalModel):
    """Build dynamax ParamsLGSSM from LinearGaussianSSM.initialize for an LTI model."""
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
        model = LinearGaussianSSM(
            state_dim=state_dim,
            emission_dim=emission_dim,
            input_dim=control_dim,
            has_dynamics_bias=evo.bias is not None,
            has_emissions_bias=obs.bias is not None,
        )
        params, _ = model.initialize(
            initial_mean=jnp.asarray(ic.loc),
            initial_covariance=jnp.asarray(ic.covariance_matrix),
            dynamics_weights=evo.A,
            dynamics_bias=evo.bias,
            dynamics_input_weights=evo.B,
            dynamics_covariance=evo.cov,
            emission_weights=obs.H,
            emission_bias=obs.bias,
            emission_input_weights=obs.D,
            emission_covariance=obs.R,
        )
        return params
    else:
        raise TypeError(
            "filter_type='kf' expects a DynamicalModel with LinearGaussianStateEvolution and LinearGaussianObservation and initial_condition as MultivariateNormal."
        )


def _prepare_inputs(dynamics, obs_values, obs_times, ctrl_times, ctrl_values):
    """Prepare emissions and inputs arrays for cd-dynamax discrete filters."""
    emissions = obs_values
    T1 = emissions.shape[0]
    control_dim = dynamics.control_dim
    if ctrl_values is None:
        inputs = jnp.zeros((T1, control_dim))
    elif ctrl_values.shape[0] > T1:
        inds = jnp.searchsorted(ctrl_times, obs_times, side="left")
        inputs = ctrl_values[inds]
    else:
        inputs = ctrl_values
    return emissions, inputs


def compute_cd_dynamax_discrete_filter(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX cd-dynamax discrete filter computation (no numpyro side-effects).

    Returns:
        PosteriorGSSMFiltered: The filtered posterior (contains marginal_loglik,
        filtered_means, filtered_covariances).
    """
    emissions, inputs = _prepare_inputs(
        dynamics, obs_values, obs_times, ctrl_times, ctrl_values
    )

    if isinstance(filter_config, KFConfig):
        params = _lti_to_lgssm_params(dynamics)
        return lgssm_filter(params, emissions, inputs=inputs)

    # EKF and UKF share the same nonlinear params representation.
    params_nl = gaussian_to_nlgssm_params(dynamics)

    if isinstance(filter_config, EKFConfig):
        return extended_kalman_filter(params_nl, emissions, inputs=inputs)
    elif isinstance(filter_config, UKFConfig):
        hyperparams = UKFHyperParams(
            alpha=filter_config.alpha,
            beta=filter_config.beta,
            kappa=filter_config.kappa,
        )
        return unscented_kalman_filter(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )
    else:
        raise ValueError(
            f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
            "Expected KFConfig, EKFConfig, or UKFConfig."
        )


def _add_kf_sites(
    name: str, posterior: PosteriorGSSMFiltered, record_kwargs: dict
) -> None:
    """Add filtered means/covariances as deterministic sites (dynamax KF posterior)."""
    max_elems = record_kwargs["record_max_elems"]
    if posterior.filtered_means is None:
        return
    means = posterior.filtered_means
    covs = posterior.filtered_covariances
    T1, state_dim = means.shape
    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], means.shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"],
        (T1, state_dim, state_dim),
        max_elems,
    )
    add_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"], (T1, state_dim), max_elems
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
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[dist.Distribution]:
    """Run discrete-time filter via cd-dynamax (KF, EKF, UKF).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: KFConfig, EKFConfig, or UKFConfig.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).

    Returns:
        list[dist.Distribution]: Filtered state distributions at each obs time.
    """
    posterior = compute_cd_dynamax_discrete_filter(
        dynamics,
        filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    record_kwargs = _config_to_record_kwargs(filter_config)
    marginal_loglik = posterior.marginal_loglik
    numpyro.factor(f"{name}_marginal_log_likelihood", marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", marginal_loglik)
    _add_kf_sites(name, posterior, record_kwargs)

    if posterior.filtered_means is None or posterior.filtered_covariances is None:
        return []
    return [
        dist.MultivariateNormal(
            posterior.filtered_means[i], posterior.filtered_covariances[i]
        )
        for i in range(posterior.filtered_means.shape[0])
    ]
