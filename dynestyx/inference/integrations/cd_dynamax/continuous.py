"""Continuous-time filters via CD-Dynamax: EnKF, DPF, EKF, UKF."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro

from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from dynestyx.dynamical_models import Context, DynamicalModel
from dynestyx.inference.integrations.cd_dynamax.utils import dsx_to_cd_dynamax
from dynestyx.utils import (
    _get_controls,
    _should_record_field,
    _validate_control_dim,
)

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM

_CONTINUOUS_FILTER_TYPES: list[str] = ["default", "enkf", "dpf", "ekf", "ukf"]


def run_continuous_filter(
    name: str,
    filter_type: str,
    dynamics: DynamicalModel,
    context: Context,
    key: jax.Array | None = None,
    filter_kwargs: dict | None = None,
    record_kwargs: dict = {},
) -> None:
    """Run continuous-time filter via CD-Dynamax.

    Args:
        name: Name of the factor.
        filter_type: Type of filter (enkf, dpf, ekf, ukf, default).
        dynamics: Dynamical model to filter.
        context: Context containing the observations and controls.
        key: Random key for the filter.
        filter_kwargs: Keyword arguments for the filter.
        record_kwargs: Keyword arguments for recording filtered states.
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    obs_traj = context.observations
    if obs_traj.times is None or obs_traj.values is None:
        return

    obs_times = obs_traj.times[:, None]
    obs_values = obs_traj.values
    ctrl_times, ctrl_values = _get_controls(context, obs_traj.times)
    _validate_control_dim(dynamics, ctrl_values)

    if filter_type.lower() in ["enkf", "default", "ekf", "ukf"]:
        cd_dynamax_model: SSMType = ContDiscreteNonlinearGaussianSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            input_dim=dynamics.control_dim,
        )

        if filter_type.lower() in ["enkf", "default"]:
            filter_type = "EnKF"

        params, _ = dsx_to_cd_dynamax(dynamics, cd_model=cd_dynamax_model)
        key = key if key is not None else jr.PRNGKey(0)

        filter_kwargs = {
            "params": params,
            "emissions": obs_values,
            "t_emissions": obs_times,
            "key": key,
            "filter_type": filter_type,
            "filter_state_order": filter_kwargs.get("filter_state_order", "first"),
            "filter_emission_order": filter_kwargs.get(
                "filter_emission_order", "first"
            ),
            "filter_num_iter": filter_kwargs.get("filter_num_iter", 1),
            "filter_state_cov_rescaling": filter_kwargs.get(
                "filter_state_cov_rescaling", 1.0
            ),
            "filter_dt_average": filter_kwargs.get("filter_dt_average", 0.1),
            "enkf_N_particles": filter_kwargs.get("enkf_N_particles", 25),
            "enkf_inflation_delta": filter_kwargs.get("enkf_inflation_delta", 0.0),
            "diffeqsolve_max_steps": filter_kwargs.get("diffeqsolve_max_steps", 1_000),
            "diffeqsolve_dt0": filter_kwargs.get("diffeqsolve_dt0", 0.01),
            "diffeqsolve_kwargs": filter_kwargs.get("diffeqsolve_kwargs", {}),
            "extra_filter_kwargs": filter_kwargs.get("extra_filter_kwargs", {}),
            "output_fields": filter_kwargs.get("output_fields", None),
            "warn": filter_kwargs.get("warn", True),
            "inputs": ctrl_values,
        }
    elif filter_type.lower() == "dpf":
        cd_dynamax_model = ContDiscreteNonlinearSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            input_dim=dynamics.control_dim,
        )

        params, _ = dsx_to_cd_dynamax(dynamics, cd_model=cd_dynamax_model)
        key = key if key is not None else jr.PRNGKey(0)

        filter_kwargs = {
            "params": params,
            "emissions": obs_values,
            "t_emissions": obs_times,
            "key": key,
            "N_particles": filter_kwargs.get("N_particles", 1_000),
            "extra_filter_kwargs": {
                "resampling_type": filter_kwargs.get("resampling_type", "stop_gradient")
            },
            "diffeqsolve_max_steps": filter_kwargs.get("diffeqsolve_max_steps", 1_000),
            "diffeqsolve_dt0": filter_kwargs.get("diffeqsolve_dt0", 0.01),
            "diffeqsolve_kwargs": filter_kwargs.get("diffeqsolve_kwargs", {}),
            "output_fields": filter_kwargs.get("output_fields", None),
            "warn": filter_kwargs.get("warn", True),
            "inputs": ctrl_values,
        }
    else:
        raise ValueError(
            f"Invalid filter type: {filter_type}. Valid types: {_CONTINUOUS_FILTER_TYPES}"
        )

    filtered = cd_dynamax_model.filter(**filter_kwargs)  # type: ignore

    numpyro.factor(f"{name}_marginal_log_likelihood", filtered.marginal_loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", filtered.marginal_loglik)

    max_elems = record_kwargs.get("record_max_elems", 100_000)
    means_shape = filtered.filtered_means.shape
    cov_shape = filtered.filtered_covariances.shape

    add_mean = _should_record_field(
        record_kwargs.get("record_filtered_states_mean"), means_shape, max_elems
    )
    add_cov = _should_record_field(
        record_kwargs.get("record_filtered_states_cov"), cov_shape, max_elems
    )
    add_cov_diag = _should_record_field(
        record_kwargs.get("record_filtered_states_cov_diag"),
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
