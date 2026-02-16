"""Continuous-time filters via CD-Dynamax: EnKF, DPF, EKF, UKF."""

import jax
import jax.numpy as jnp
import numpyro

from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from dynestyx.dynamical_models import Context, DynamicalModel
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeUKFConfig,
    config_to_record_kwargs,
)
from dynestyx.inference.integrations.cd_dynamax.utils import dsx_to_cd_dynamax
from dynestyx.utils import (
    _get_controls,
    _should_record_field,
    _validate_control_dim,
)

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM

ContinuousTimeFilterConfig = (
    ContinuousTimeEnKFConfig
    | ContinuousTimeDPFConfig
    | ContinuousTimeEKFConfig
    | ContinuousTimeUKFConfig
)


def _config_to_cd_dynamax_filter_kwargs(
    config: ContinuousTimeFilterConfig,
    params,
    obs_values,
    obs_times,
    ctrl_values,
    key,
) -> dict:
    """Build the filter_kwargs dict passed to cd_dynamax_model.filter()."""
    base = {
        "params": params,
        "emissions": obs_values,
        "t_emissions": obs_times,
        "inputs": ctrl_values,
        "key": key,
        "filter_state_order": config.filter_state_order,
        "filter_state_cov_rescaling": config.cov_rescaling
        if config.cov_rescaling is not None
        else 1.0,
        "diffeqsolve_max_steps": config.diffeqsolve_max_steps,
        "diffeqsolve_dt0": config.diffeqsolve_dt0,
        "diffeqsolve_kwargs": config.diffeqsolve_kwargs,
        "extra_filter_kwargs": config.extra_filter_kwargs,
        "warn": config.warn,
    }
    if isinstance(config, ContinuousTimeEnKFConfig):
        base["filter_type"] = "EnKF"
        base["enkf_N_particles"] = config.n_particles
        base["enkf_inflation_delta"] = (
            config.inflation_delta if config.inflation_delta is not None else 0.0
        )
        base["extra_filter_kwargs"] = {
            "perturb_measurements": config.perturb_measurements
            if config.perturb_measurements is not None
            else True
        }
    elif isinstance(config, ContinuousTimeEKFConfig):
        base["filter_type"] = "EKF"
        base["filter_emission_order"] = config.filter_emission_order
        base["filter_num_iter"] = 1
    elif isinstance(config, ContinuousTimeUKFConfig):
        base["filter_type"] = "UKF"
        base["extra_filter_kwargs"] = {
            "alpha": config.alpha,
            "beta": config.beta,
            "kappa": config.kappa,
        }
    elif isinstance(config, ContinuousTimeDPFConfig):
        if config.resampling_method.base_method != "multinomial":
            raise ValueError(
                "Only multinomial resampling is supported for CD-Dynamax DPF."
            )
        base["filter_type"] = "DPF"
        base["N_particles"] = config.n_particles
        base["extra_filter_kwargs"] = {
            "resample_method": config.resampling_method.differential_method,
            "softness": config.resampling_method.softness,
            "ess_threshold_ratio": config.ess_threshold_ratio,
        }
    return base


def run_continuous_filter(
    name: str,
    dynamics: DynamicalModel,
    context: Context,
    filter_config: ContinuousTimeFilterConfig,
    key: jax.Array | None = None,
) -> None:
    """Run continuous-time filter via CD-Dynamax.

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        context: Context containing the observations and controls.
        filter_config: ContinuousTimeEnKFConfig, ContinuousTimeDPFConfig,
            ContinuousTimeEKFConfig, or ContinuousTimeUKFConfig.
        key: Random key (optional).
    """

    obs_traj = context.observations
    if obs_traj.times is None or obs_traj.values is None:
        return

    if isinstance(filter_config, (ContinuousTimeEnKFConfig, ContinuousTimeDPFConfig)):
        if key is None:
            raise ValueError(
                f"{type(filter_config).__name__} requires a PRNG key: set 'crn_seed' in the filter config, "
                "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
            )

    obs_times = obs_traj.times[:, None]
    obs_values = obs_traj.values
    ctrl_times, ctrl_values = _get_controls(context, obs_traj.times)
    _validate_control_dim(dynamics, ctrl_values)

    if isinstance(filter_config, ContinuousTimeDPFConfig):
        cd_dynamax_model: SSMType = ContDiscreteNonlinearSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            input_dim=dynamics.control_dim,
        )
    else:
        cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            input_dim=dynamics.control_dim,
        )

    params, _ = dsx_to_cd_dynamax(dynamics, cd_model=cd_dynamax_model)
    filter_kwargs = _config_to_cd_dynamax_filter_kwargs(
        filter_config, params, obs_values, obs_times, ctrl_values, key
    )

    filtered = cd_dynamax_model.filter(**filter_kwargs)  # type: ignore

    record_kwargs = config_to_record_kwargs(filter_config)

    numpyro.factor(f"{name}_marginal_log_likelihood", filtered.marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", filtered.marginal_loglik)

    max_elems = record_kwargs["record_max_elems"]
    means_shape = filtered.filtered_means.shape
    cov_shape = filtered.filtered_covariances.shape

    add_mean = _should_record_field(
        record_kwargs["record_filtered_states_mean"], means_shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs["record_filtered_states_cov"], cov_shape, max_elems
    )
    add_cov_diag = _should_record_field(
        record_kwargs["record_filtered_states_cov_diag"],
        (cov_shape[0], cov_shape[1]),
        max_elems,
    )

    if add_mean:
        numpyro.deterministic(f"{name}_filtered_states_mean", filtered.filtered_means)
    if add_cov:
        numpyro.deterministic(
            f"{name}_filtered_states_cov", filtered.filtered_covariances
        )
    if add_cov_diag:
        diag_cov = jnp.diagonal(filtered.filtered_covariances, axis1=1, axis2=2)
        numpyro.deterministic(f"{name}_filtered_states_cov_diag", diag_cov)
