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
    """Build dynamax ParamsLGSSM via builders.build_params from an LTI model."""
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
    record_kwargs: dict,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[dist.Distribution]:
    """Run dynamax Kalman filter for LTI_discretetime and add factor + sites."""
    emissions = obs_values
    T1 = emissions.shape[0]
    control_dim = dynamics.control_dim
    if ctrl_values is None:
inputs = jnp.zeros((T1, control_dim))
    elif ctrl_values.shape[0] > T1:
        # Find controls aligned to obs_times
        inds = jnp.searchsorted(ctrl_times, obs_times, side="left")
        inputs = ctrl_values[inds]
    else:
        # Controls should align exactly with obs_times
        inputs = ctrl_values

    params = _lti_to_lgssm_params(dynamics)
    posterior = lgssm_filter(params, emissions, inputs=inputs)

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


def _run_nlgssm_filter(
    name: str,
    dynamics: DynamicalModel,
    record_kwargs: dict,
    run_filter: Callable,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[dist.Distribution]:
    """Common setup for EKF/UKF: get emissions/inputs, run filter, add factor + sites."""
    emissions = obs_values
    T1 = emissions.shape[0]
    control_dim = dynamics.control_dim
    if ctrl_values is None:
        inputs = jnp.zeros((T1, control_dim))
    elif ctrl_values.shape[0] > T1:
        # Find controls aligned to obs_times
        inds = jnp.searchsorted(ctrl_times, obs_times, side="left")
        inputs = ctrl_values[inds]
    else:
        # Controls should align exactly with obs_times
        inputs = ctrl_values

    params_nl = gaussian_to_nlgssm_params(dynamics)
    posterior = run_filter(params_nl, emissions, inputs)

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


def _filter_discrete_time_dynamax_ekf(
    name: str,
    dynamics: DynamicalModel,
    record_kwargs: dict,
    num_iter: int = 1,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[dist.Distribution]:
    """Run dynamax EKF for LTI and add factor + sites."""

    def run_filter(params_nl, emissions, inputs):
        return extended_kalman_filter(
            params_nl, emissions, num_iter=num_iter, inputs=inputs
        )

    return _run_nlgssm_filter(
        name,
        dynamics,
        record_kwargs,
        run_filter,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        **kwargs,
    )


def _filter_discrete_time_dynamax_ukf(
    name: str,
    dynamics: DynamicalModel,
    record_kwargs: dict,
    hyperparams: UKFHyperParams | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[dist.Distribution]:
    """Run dynamax UKF for LTI and add factor + sites."""
    if hyperparams is None:
        hyperparams = UKFHyperParams()

    def run_filter(params_nl, emissions, inputs):
        return unscented_kalman_filter(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )

    return _run_nlgssm_filter(
        name,
        dynamics,
        record_kwargs,
        run_filter,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        **kwargs,
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

    Returns:
        list[dist.Distribution]: Filtered state distributions at each obs time.
    """
    record_kwargs = _config_to_record_kwargs(filter_config)
    filter_kwargs = dict(
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        **kwargs,
    )

    if isinstance(filter_config, KFConfig):
        return _filter_discrete_time_dynamax_kf(
            name, dynamics, record_kwargs, **filter_kwargs
        )
    elif isinstance(filter_config, EKFConfig):
        return _filter_discrete_time_dynamax_ekf(
            name, dynamics, record_kwargs, **filter_kwargs
        )
    elif isinstance(filter_config, UKFConfig):
        hyperparams = UKFHyperParams(
            alpha=filter_config.alpha,
            beta=filter_config.beta,
            kappa=filter_config.kappa,
        )
        return _filter_discrete_time_dynamax_ukf(
            name, dynamics, record_kwargs, hyperparams=hyperparams, **filter_kwargs
        )
    else:
        raise ValueError(
            f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
            "Expected KFConfig, EKFConfig, or UKFConfig."
        )
