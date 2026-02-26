"""Continuous-time filters via CD-Dynamax: KF, EnKF, DPF, EKF, UKF."""

import jax
import jax.numpy as jnp
import numpyro

from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
)
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.models import (
    PosteriorGSSMFiltered,
)
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
    _config_to_record_kwargs,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    dsx_to_cd_dynamax,
    dsx_to_cdlgssm_params,
)
from dynestyx.models import DynamicalModel
from dynestyx.utils import (
    _should_record_field,
    _validate_control_dim,
    _validate_controls,
    _validate_predict_times,
)

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM

ContinuousTimeFilterConfig = (
    ContinuousTimeKFConfig
    | ContinuousTimeEnKFConfig
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
            else True,
            **config.extra_filter_kwargs,
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
            **config.extra_filter_kwargs,
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
            **config.extra_filter_kwargs,
        }
    return base


def _add_filter_sites(
    name: str,
    filter_config: ContinuousTimeFilterConfig,
    filtered,
) -> None:
    """Add marginal log-likelihood factor and filtered state deterministic sites."""
    record_kwargs = _config_to_record_kwargs(filter_config)
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


def _run_linear_kf(
    name: str,
    dynamics: DynamicalModel,
    obs_times,
    obs_values,
    ctrl_values,
    predict_times,
    filter_config: ContinuousTimeKFConfig,
) -> PosteriorGSSMFiltered | tuple[PosteriorGSSMFiltered, object]:
    """Run exact continuous-discrete KF (AffineLinearDrift + constant diffusion + LinearGaussianObservation)."""
    params = dsx_to_cdlgssm_params(dynamics)
    cd_model = ContDiscreteLinearGaussianSSM(
        state_dim=dynamics.state_dim,
        emission_dim=dynamics.observation_dim,
        input_dim=dynamics.control_dim,
    )
    if predict_times is not None and len(predict_times) > 0:
        filtered, forecasted = cd_model.filter_and_forecast(
            params=params,
            emissions_filter=obs_values,
            t_emissions_filter=obs_times,
            t_emissions_forecast=predict_times,
            inputs_filter=ctrl_values,
            inputs_forecast=None,
            warn=filter_config.warn,
        )
        return filtered, forecasted

    filtered = cd_model.filter(
        params=params,
        emissions=obs_values,
        t_emissions=obs_times,
        inputs=ctrl_values,
        warn=filter_config.warn,
    )
    return filtered


def run_continuous_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    predict_times: jax.Array | None = None,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> None:
    """Run continuous-time filter via CD-Dynamax.

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: ContinuousTimeKFConfig, ContinuousTimeEnKFConfig,
            ContinuousTimeDPFConfig, ContinuousTimeEKFConfig, or ContinuousTimeUKFConfig.
        key: Random key (optional).
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).
    """
    obs_times_arr = jnp.asarray(obs_times)
    if obs_times_arr.ndim == 1:
        obs_times_arr = obs_times_arr[:, None]
    predict_times_arr = None
    if predict_times is not None:
        predict_times_arr = jnp.asarray(predict_times)
        if predict_times_arr.ndim == 1:
            predict_times_arr = predict_times_arr[:, None]

    _validate_predict_times(
        jnp.ravel(obs_times_arr),
        None if predict_times_arr is None else jnp.ravel(predict_times_arr),
    )
    _validate_controls(jnp.ravel(obs_times_arr), ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)

    control_dim = dynamics.control_dim
    ctrl_vals = (
        ctrl_values
        if ctrl_values is not None
        else jnp.zeros((obs_times_arr.shape[0], control_dim))
    )

    if isinstance(filter_config, ContinuousTimeKFConfig):
        kf_output = _run_linear_kf(
            name,
            dynamics,
            obs_times_arr,
            obs_values,
            ctrl_vals,
            predict_times_arr,
            filter_config,
        )
        if isinstance(kf_output, tuple):
            filtered, forecasted = kf_output
        else:
            filtered = kf_output
            forecasted = None
    else:
        if isinstance(
            filter_config, (ContinuousTimeEnKFConfig, ContinuousTimeDPFConfig)
        ):
            if key is None:
                raise ValueError(
                    f"{type(filter_config).__name__} requires a PRNG key: set 'crn_seed' in the filter config, "
                    "or run inside a NumPyro seeded context (e.g., with numpyro.handlers.seed)."
                )

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
        if predict_times_arr is not None and len(predict_times_arr) > 0:
            if not hasattr(cd_dynamax_model, "filter_and_forecast"):
                raise ValueError(
                    "predict_times is not supported for this CD-Dynamax model "
                    f"({type(cd_dynamax_model).__name__}). "
                    "Only CDNLGSSM/CDLGSSM backends currently support forecasting."
                )
            if isinstance(filter_config, ContinuousTimeDPFConfig):
                raise ValueError(
                    "predict_times is not supported for ContinuousTimeDPFConfig "
                    "(CDNLSSM backend has no filter_and_forecast)."
                )
            filter_kwargs = _config_to_cd_dynamax_filter_kwargs(
                filter_config, params, obs_values, obs_times_arr, ctrl_vals, key
            )
            filtered, forecasted = cd_dynamax_model.filter_and_forecast(  # type: ignore[attr-defined]
                params=filter_kwargs["params"],
                emissions_filter=filter_kwargs["emissions"],
                t_emissions_filter=filter_kwargs["t_emissions"],
                t_emissions_forecast=predict_times_arr,
                inputs_filter=filter_kwargs["inputs"],
                inputs_forecast=None,
                filter_type=filter_kwargs["filter_type"],
                filter_state_order=filter_kwargs["filter_state_order"],
                filter_emission_order=filter_kwargs.get(
                    "filter_emission_order", "first"
                ),
                filter_num_iter=filter_kwargs.get("filter_num_iter", 1),
                filter_state_cov_rescaling=filter_kwargs["filter_state_cov_rescaling"],
                enkf_N_particles=filter_kwargs.get("enkf_N_particles", 25),
                enkf_inflation_delta=filter_kwargs.get("enkf_inflation_delta", 0.0),
                diffeqsolve_max_steps=filter_kwargs["diffeqsolve_max_steps"],
                diffeqsolve_dt0=filter_kwargs["diffeqsolve_dt0"],
                key=filter_kwargs["key"],
                diffeqsolve_kwargs=filter_kwargs["diffeqsolve_kwargs"],
                extra_filter_kwargs=filter_kwargs["extra_filter_kwargs"],
                warn=filter_kwargs["warn"],
            )
        else:
            filter_kwargs = _config_to_cd_dynamax_filter_kwargs(
                filter_config, params, obs_values, obs_times_arr, ctrl_vals, key
            )
            filtered = cd_dynamax_model.filter(**filter_kwargs)  # type: ignore
            forecasted = None

    _add_filter_sites(name, filter_config, filtered)
    if forecasted is not None:
        numpyro.deterministic(
            f"{name}_forecasted_state_means", forecasted.forecasted_state_means
        )
        numpyro.deterministic(
            f"{name}_forecasted_state_covs", forecasted.forecasted_state_covariances
        )
