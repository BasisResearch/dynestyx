"""Discrete-time smoothers via cd-dynamax (dynamax): KF, EKF, UKF."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_smoother
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_smoother,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    unscented_kalman_smoother,
)

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    EKFConfig,
    KFConfig,
    UKFConfig,
    _config_to_smoother_record_kwargs,
)
from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (
    _lti_to_lgssm_params,
    _prepare_inputs,
)
from dynestyx.inference.integrations.cd_dynamax.utils import gaussian_to_nlgssm_params
from dynestyx.models import DynamicalModel
from dynestyx.utils import _should_record_field


def compute_cd_dynamax_discrete_smoother(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX cd-dynamax discrete smoother computation (no numpyro side-effects)."""
    emissions, inputs = _prepare_inputs(
        dynamics, obs_values, obs_times, ctrl_times, ctrl_values
    )

    if isinstance(filter_config, KFConfig):
        params = _lti_to_lgssm_params(dynamics)
        return lgssm_smoother(params, emissions, inputs=inputs)

    params_nl = gaussian_to_nlgssm_params(dynamics)
    if isinstance(filter_config, EKFConfig):
        return extended_kalman_smoother(params_nl, emissions, inputs=inputs)
    if isinstance(filter_config, UKFConfig):
        hyperparams = UKFHyperParams(
            alpha=filter_config.alpha,
            beta=filter_config.beta,
            kappa=filter_config.kappa,
        )
        return unscented_kalman_smoother(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )
    raise ValueError(
        f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
        "Expected KFConfig, EKFConfig, or UKFConfig."
    )


def _add_smoother_sites(name: str, posterior, record_kwargs: dict) -> None:
    """Add smoothed means/covariances as deterministic sites."""
    max_elems = record_kwargs["record_max_elems"]
    means = posterior.smoothed_means
    covs = posterior.smoothed_covariances
    if means is None or covs is None:
        return
    t1, state_dim = means.shape
    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], means.shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs["record_smoothed_states_cov"],
        (t1, state_dim, state_dim),
        max_elems,
    )
    add_cov_diag = _should_record_field(
        record_kwargs["record_smoothed_states_cov_diag"], (t1, state_dim), max_elems
    )
    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_states_mean", means)
    if add_cov:
        numpyro.deterministic(f"{name}_smoothed_states_cov", covs)
    if add_cov_diag:
        diag_cov = jnp.diagonal(covs, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_smoothed_states_cov_diag", diag_cov)


def run_discrete_smoother(
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
    """Run discrete-time smoother via cd-dynamax (KF, EKF, UKF)."""
    posterior = compute_cd_dynamax_discrete_smoother(
        dynamics,
        filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    numpyro.factor(f"{name}_marginal_log_likelihood", posterior.marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", posterior.marginal_loglik)
    _add_smoother_sites(
        name, posterior, _config_to_smoother_record_kwargs(filter_config)
    )

    means = posterior.smoothed_means
    covs = posterior.smoothed_covariances
    if means is None or covs is None:
        return []
    return [dist.MultivariateNormal(means[i], covs[i]) for i in range(means.shape[0])]


__all__ = ["compute_cd_dynamax_discrete_smoother", "run_discrete_smoother"]
