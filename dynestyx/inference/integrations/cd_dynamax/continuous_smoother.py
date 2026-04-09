"""Continuous-time smoothers via CD-Dynamax: KF, EKF."""

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from cd_dynamax import (
    ContDiscreteNonlinearGaussianSSM,
    EKFHyperParams,
    KFHyperParams,
    cdlgssm_smoother,
    cdnlgssm_smoother,
)

from dynestyx.inference.filter_configs import _config_to_smoother_record_kwargs
from dynestyx.inference.integrations.cd_dynamax.utils import (
    dsx_to_cd_dynamax,
    dsx_to_cdlgssm_params,
)
from dynestyx.inference.smoother_configs import (
    ContinuousTimeEKFSmootherConfig,
    ContinuousTimeKFSmootherConfig,
)
from dynestyx.models import DynamicalModel
from dynestyx.utils import _should_record_field

ContinuousTimeSmootherConfig = (
    ContinuousTimeKFSmootherConfig | ContinuousTimeEKFSmootherConfig
)


def _add_smoother_sites(
    name: str,
    smoother_config: ContinuousTimeSmootherConfig,
    smoothed,
) -> None:
    """Add marginal log-likelihood factor and smoothed state deterministic sites."""
    record_kwargs = _config_to_smoother_record_kwargs(smoother_config)
    numpyro.factor(f"{name}_marginal_log_likelihood", smoothed.marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", smoothed.marginal_loglik)

    max_elems = record_kwargs["record_max_elems"]
    means = smoothed.smoothed_means
    covs = smoothed.smoothed_covariances
    means_shape = means.shape
    cov_shape = covs.shape
    add_mean = _should_record_field(
        record_kwargs["record_smoothed_states_mean"], means_shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs["record_smoothed_states_cov"], cov_shape, max_elems
    )
    add_cov_diag = _should_record_field(
        record_kwargs["record_smoothed_states_cov_diag"],
        (cov_shape[0], cov_shape[1]),
        max_elems,
    )
    if add_mean:
        numpyro.deterministic(f"{name}_smoothed_states_mean", means)
    if add_cov:
        numpyro.deterministic(f"{name}_smoothed_states_cov", covs)
    if add_cov_diag:
        diag_cov = jnp.diagonal(covs, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_smoothed_states_cov_diag", diag_cov)


def compute_continuous_smoother(
    dynamics: DynamicalModel,
    smoother_config: ContinuousTimeSmootherConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX continuous-time smoother computation (no numpyro side-effects)."""
    obs_times_arr = jnp.asarray(obs_times)
    if obs_times_arr.ndim == 1:
        obs_times_arr = obs_times_arr[:, None]

    control_dim = dynamics.control_dim
    ctrl_vals = (
        ctrl_values
        if ctrl_values is not None
        else jnp.zeros((obs_times_arr.shape[0], control_dim))
    )

    if isinstance(smoother_config, ContinuousTimeKFSmootherConfig):
        params = dsx_to_cdlgssm_params(dynamics)
        kf_hparams = KFHyperParams(
            dt_final=float(smoother_config.extra_filter_kwargs.get("dt_final", 1e-10)),
            diffeqsolve_settings={
                "dt0": smoother_config.diffeqsolve_dt0,
                "max_steps": smoother_config.diffeqsolve_max_steps,
                **smoother_config.diffeqsolve_kwargs,
            },
        )
        return cdlgssm_smoother(
            params=params,
            emissions=obs_values,
            t_emissions=obs_times_arr,
            filter_hyperparams=kf_hparams,
            inputs=ctrl_vals,
            smoother_type=smoother_config.cdlgssm_smoother_type,
            warn=smoother_config.warn,
        )

    if isinstance(smoother_config, ContinuousTimeEKFSmootherConfig):
        cd_model = ContDiscreteNonlinearGaussianSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            input_dim=dynamics.control_dim,
        )
        params, _ = dsx_to_cd_dynamax(dynamics, cd_model=cd_model)
        ekf_hparams = EKFHyperParams(
            dt_final=float(smoother_config.extra_filter_kwargs.get("dt_final", 1e-4)),
            state_order=smoother_config.filter_state_order,
            emission_order=smoother_config.filter_emission_order,
            smooth_order=str(
                smoother_config.extra_filter_kwargs.get("smooth_order", "first")
            ),
            cov_rescaling=(
                smoother_config.cov_rescaling
                if smoother_config.cov_rescaling is not None
                else 1.0
            ),
            diffeqsolve_settings={
                "dt0": smoother_config.diffeqsolve_dt0,
                "max_steps": smoother_config.diffeqsolve_max_steps,
                **smoother_config.diffeqsolve_kwargs,
            },
            dt_average=float(
                smoother_config.extra_filter_kwargs.get("dt_average", 0.1)
            ),
        )
        return cdnlgssm_smoother(
            params=params,
            emissions=obs_values,
            t_emissions=obs_times_arr,
            filter_hyperparams=ekf_hparams,
            inputs=ctrl_vals,
            num_iter=1,
            key=key if key is not None else jax.random.PRNGKey(0),
            warn=smoother_config.warn,
        )

    raise ValueError(
        f"{type(smoother_config).__name__} smoothing is not supported in cd_dynamax. "
        "Supported continuous-time smoothers: ContinuousTimeKFSmootherConfig, ContinuousTimeEKFSmootherConfig."
    )


def run_continuous_smoother(
    name: str,
    dynamics: DynamicalModel,
    smoother_config: ContinuousTimeSmootherConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Run continuous-time smoother via CD-Dynamax."""
    smoothed = compute_continuous_smoother(
        dynamics,
        smoother_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    _add_smoother_sites(name, smoother_config, smoothed)
    means = smoothed.smoothed_means
    covs = smoothed.smoothed_covariances
    return [dist.MultivariateNormal(means[i], covs[i]) for i in range(means.shape[0])]


__all__ = ["compute_continuous_smoother", "run_continuous_smoother"]
