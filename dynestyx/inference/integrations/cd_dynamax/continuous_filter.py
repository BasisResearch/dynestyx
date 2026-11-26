"""Continuous-time filters via CD-Dynamax: KF, EnKF, DPF, EKF, UKF."""

from typing import Any

import numpyro.distributions as dist
from cd_dynamax import (
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
)
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.inference import (
    KFHyperParams,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.models import (
    build_filter_hyperparams,
)
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.inference_dpf import (
    build_dpf_hyperparams,
)
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.configs.filters import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
)
from dynestyx.inference.custom_filter import step_inputs
from dynestyx.inference.integrations.cd_dynamax.continuous_steps import (
    ensemble_filter,
    linear_filter,
    nonlinear_filter,
    particle_filter,
)
from dynestyx.inference.integrations.cd_dynamax.filter_steps import CDFilter
from dynestyx.inference.integrations.cd_dynamax.utils import (
    dsx_to_cd_dynamax,
    dsx_to_cdlgssm_params,
)
from dynestyx.inference.observation_predictions import (
    wants_observation_prediction_diagnostics,
)
from dynestyx.inference.utils.distribution_utils import _posterior_sequence_to_dists
from dynestyx.models import DynamicalModel

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM

ContinuousTimeFilterConfig = (
    ContinuousTimeKFConfig
    | ContinuousTimeEnKFConfig
    | ContinuousTimeDPFConfig
    | ContinuousTimeEKFConfig
    | ContinuousTimeUKFConfig
)


def _hyperparams(config: ContinuousTimeFilterConfig):
    base = {
        "filter_state_order": config.filter_state_order,
        "filter_state_cov_rescaling": config.cov_rescaling
        if config.cov_rescaling is not None
        else 1.0,
        "diffeqsolve_max_steps": config.diffeqsolve_max_steps,
        "diffeqsolve_dt0": config.diffeqsolve_dt0,
        "diffeqsolve_kwargs": config.diffeqsolve_kwargs,
        "extra_filter_kwargs": config.extra_filter_kwargs,
    }
    if isinstance(config, ContinuousTimeEnKFConfig):
        base["filter_type"] = "EnKF"
        base["enkf_N_particles"] = config.n_particles
        base["enkf_inflation_delta"] = config.inflation_delta
        base["extra_filter_kwargs"] = {
            "perturb_measurements": config.perturb_measurements
            if config.perturb_measurements is not None
            else True,
            **config.extra_filter_kwargs,
        }
    elif isinstance(config, ContinuousTimeEKFConfig):
        base["filter_type"] = "EKF"
        base["filter_emission_order"] = config.filter_emission_order
    elif isinstance(config, ContinuousTimeUKFConfig):
        base["filter_type"] = "UKF"
        base["extra_filter_kwargs"] = {
            "alpha": config.alpha,
            "beta": config.beta,
            "kappa": config.kappa,
            **config.extra_filter_kwargs,
        }
    elif isinstance(config, ContinuousTimeDPFConfig):
        if config.resampling_method.base_method != "multinomial":
            raise ValueError(
                "Only multinomial resampling is supported for CD-Dynamax DPF."
            )
        base["N_particles"] = config.n_particles
        base["extra_filter_kwargs"] = {
            "resample_method": config.resampling_method.differential_method,
            "softness": config.resampling_method.softness,
            "ess_threshold_ratio": config.ess_threshold_ratio,
            **config.extra_filter_kwargs,
        }
        return build_dpf_hyperparams(**base)
    return build_filter_hyperparams(**base)


def compute_continuous_filter(
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
) -> Any:
    """Pure-JAX continuous-time filter computation (no numpyro side-effects)."""
    inputs = step_inputs(dynamics, obs_times, obs_values, ctrl_times, ctrl_values)
    if key is None and isinstance(
        filter_config, (ContinuousTimeEnKFConfig, ContinuousTimeDPFConfig)
    ):
        raise ValueError(
            f"{type(filter_config).__name__} requires a PRNG key: set 'crn_seed' in the filter config, or run inside a NumPyro seeded context."
        )
    return filter_config.build_filter(dynamics).run(inputs, key)


def build_continuous_filter(
    dynamics: DynamicalModel, filter_config: ContinuousTimeFilterConfig
) -> CDFilter:
    diagnostics = wants_observation_prediction_diagnostics(filter_config)
    if isinstance(filter_config, ContinuousTimeKFConfig):
        params = dsx_to_cdlgssm_params(dynamics)
        hp = KFHyperParams(
            diffeqsolve_settings={
                "dt0": filter_config.diffeqsolve_dt0,
                "max_steps": filter_config.diffeqsolve_max_steps,
                **filter_config.diffeqsolve_kwargs,
            }
        )
        return linear_filter(
            params, hp, diagnostics, filter_config.warn, dynamics.control_dim
        )

    model_cls = (
        ContDiscreteNonlinearSSM
        if isinstance(filter_config, ContinuousTimeDPFConfig)
        else ContDiscreteNonlinearGaussianSSM
    )
    model = model_cls(
        state_dim=dynamics.state_dim,
        emission_dim=dynamics.observation_dim,
        input_dim=dynamics.control_dim,
    )
    params, _ = dsx_to_cd_dynamax(dynamics, cd_model=model)
    hp = _hyperparams(filter_config)
    if isinstance(filter_config, ContinuousTimeDPFConfig):
        return particle_filter(params, hp)
    if isinstance(filter_config, ContinuousTimeEnKFConfig):
        return ensemble_filter(params, hp, diagnostics, filter_config.warn)
    return nonlinear_filter(params, hp, diagnostics, filter_config.warn)


def run_continuous_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
    **kwargs,
) -> tuple[Real[Array, ""], object, list[dist.Distribution]]:
    """Run continuous-time filter via CD-Dynamax.

    Pure computation — no numpyro side-effects. Callers are responsible for
    registering numpyro.factor / numpyro.deterministic if needed.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}).
            - filtered_posterior: CD-Dynamax posterior object with filtered_means,
              filtered_covariances, and marginal_loglik attributes.
            - filtered_dists: list of MultivariateNormal distributions p(x_t | y_{1:t})
              at each obs time, for posterior rollout.
    """
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    filtered_dists = _posterior_sequence_to_dists(
        filtered,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=isinstance(filter_config, ContinuousTimeDPFConfig),
        missing_message="Filtered means/covariances unexpectedly None for non-DPF config",
    )
    return filtered.marginal_loglik, filtered, filtered_dists


__all__ = [
    "ContinuousTimeFilterConfig",
    "compute_continuous_filter",
    "run_continuous_filter",
]
