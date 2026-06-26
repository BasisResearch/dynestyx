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

from dynestyx.inference.distribution_utils import _posterior_sequence_to_dists
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
from dynestyx.inference.observation_predictions import (
    enrich_and_record_continuous_filter_output,
    wants_observation_prediction_diagnostics,
)
from dynestyx.inference.scoring_configs import ObservationScoringConfig
from dynestyx.models import DynamicalModel
from dynestyx.utils import _should_record_field

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
    output_fields,
) -> dict:
    """Build the filter_kwargs dict passed to cd_dynamax_model.filter()."""

    # cd-dynamax uses the legacy PRNG key interface, but newer numpyro uses typed keys.
    # We should convert accordingly.
    # https://docs.jax.dev/en/latest/jax.random.html#module-jax.random
    if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
        key = jax.random.key_data(key)

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
        "output_fields": output_fields,
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


def _continuous_filter_output_fields(
    filter_config: ContinuousTimeFilterConfig,
    *,
    scoring_config: ObservationScoringConfig | None,
) -> list[str] | None:
    """Select the CD-Dynamax posterior fields Dynestyx needs for this run."""
    if isinstance(filter_config, ContinuousTimeDPFConfig):
        return None

    output_fields = [
        "marginal_loglik",
        "filtered_means",
        "filtered_covariances",
    ]
    if not wants_observation_prediction_diagnostics(
        filter_config,
        scoring_config=scoring_config,
    ):
        return output_fields

    output_fields.extend(
        [
            "y_pred_mean",
            "y_pred_cov",
            "y_obs_pred_mean",
            "y_obs_pred_cov",
        ]
    )
    if isinstance(filter_config, ContinuousTimeEnKFConfig):
        output_fields.extend(
            [
                "y_ens_pred",
                "y_obs_ens_pred",
            ]
        )
    return output_fields


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
    dynamics: DynamicalModel,
    obs_times,
    obs_values,
    ctrl_values,
    filter_config: ContinuousTimeKFConfig,
    *,
    output_fields: list[str] | None,
) -> PosteriorGSSMFiltered:
    """Run exact continuous-discrete KF (AffineLinearDrift + constant diffusion + LinearGaussianObservation)."""
    params = dsx_to_cdlgssm_params(dynamics)
    cd_model = ContDiscreteLinearGaussianSSM(
        state_dim=dynamics.state_dim,
        emission_dim=dynamics.observation_dim,
        input_dim=dynamics.control_dim,
    )
    filtered = cd_model.filter(
        params=params,
        emissions=obs_values,
        t_emissions=obs_times,
        inputs=ctrl_values,
        output_fields=output_fields,
        warn=filter_config.warn,
    )
    return filtered


def compute_continuous_filter(
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    scoring_config: ObservationScoringConfig | None = None,
):
    """Pure-JAX continuous-time filter computation (no numpyro side-effects)."""
    obs_times_arr = jnp.asarray(obs_times)
    if obs_times_arr.ndim == 1:
        obs_times_arr = obs_times_arr[:, None]

    control_dim = dynamics.control_dim
    ctrl_vals = (
        ctrl_values
        if ctrl_values is not None
        else jnp.zeros((obs_times_arr.shape[0], control_dim))
    )

    output_fields = _continuous_filter_output_fields(
        filter_config,
        scoring_config=scoring_config,
    )

    if isinstance(filter_config, ContinuousTimeKFConfig):
        filtered = _run_linear_kf(
            dynamics,
            obs_times_arr,
            obs_values,
            ctrl_vals,
            filter_config,
            output_fields=output_fields,
        )
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
        filter_kwargs = _config_to_cd_dynamax_filter_kwargs(
            filter_config,
            params,
            obs_values,
            obs_times_arr,
            ctrl_vals,
            key,
            output_fields,
        )

        filtered = cd_dynamax_model.filter(**filter_kwargs)  # type: ignore

    return filtered


def run_continuous_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    scoring_config: ObservationScoringConfig | None = None,
    **kwargs,
) -> list[numpyro.distributions.Distribution]:
    """Run continuous-time filter via CD-Dynamax."""
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )

    filtered = enrich_and_record_continuous_filter_output(
        name,
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )

    _add_filter_sites(name, filter_config, filtered)

    return _posterior_sequence_to_dists(
        filtered,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=isinstance(filter_config, ContinuousTimeDPFConfig),
        missing_message="Filtered means/covariances unexpectedly None for non-DPF config",
    )


__all__ = [
    "ContinuousTimeFilterConfig",
    "compute_continuous_filter",
    "run_continuous_filter",
]
