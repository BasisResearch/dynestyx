"""Canonical predicted-observation summaries produced by filters.

These utilities translate backend-specific predictive-observation fields into a
small Dynestyx-level representation used by downstream handlers.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float, Real

from dynestyx.inference.configs.filter import (
    BaseFilterConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
)
from dynestyx.models import DynamicalModel
from dynestyx.models.observations import GaussianObservation, LinearGaussianObservation
from dynestyx.utils import _array_has_plate_dims, _should_record_field

type SupportedObservationPredictionConfig = (
    ContinuousTimeKFConfig
    | ContinuousTimeEKFConfig
    | ContinuousTimeUKFConfig
    | ContinuousTimeEnKFConfig
)


@dataclasses.dataclass(frozen=True)
class PredictedObservationOutputs:
    """Canonical predicted-observation outputs for Dynestyx filters."""

    mean: Float[Array, "*plate time observation_dim"] | None = None
    cov: Float[Array, "*plate time observation_dim observation_dim"] | None = None
    obs_cov: Float[Array, "*plate time observation_dim observation_dim"] | None = None
    ensemble: Float[Array, "*plate time n_members observation_dim"] | None = None
    obs_ensemble: Float[Array, "*plate time n_members observation_dim"] | None = None
    noise_cov: Float[Array, "*plate time observation_dim observation_dim"] | None = None


def _canonicalize_observations(
    arr: Float[Array, ...],
    *,
    plate_shapes: tuple[int, ...],
) -> Float[Array, ...]:
    time_axis = len(plate_shapes)
    if arr.ndim == time_axis + 1:
        return arr[..., None]
    return arr


def _observation_control_values(
    dynamics: DynamicalModel,
    *,
    obs_times: Real[Array, "... time"],
    ctrl_values: Real[Array, "... control_time control_dim"]
    | Real[Array, "... control_time"]
    | None,
    plate_shapes: tuple[int, ...],
) -> Real[Array, "... control_time control_dim"] | None:
    if dynamics.control_dim == 0:
        return None
    if ctrl_values is None:
        t_len = int(jnp.asarray(obs_times).shape[-1])
        return jnp.zeros(
            (*plate_shapes, t_len, dynamics.control_dim), dtype=obs_times.dtype
        )

    ctrl_arr = jnp.asarray(ctrl_values)
    has_plate_dims = _array_has_plate_dims(
        ctrl_arr,
        plate_shapes,
        min_suffix_ndim=1,
    )
    if has_plate_dims and ctrl_arr.ndim == len(plate_shapes) + 1:
        ctrl_arr = ctrl_arr[..., None]
    elif not has_plate_dims:
        if ctrl_arr.ndim == 1:
            ctrl_arr = ctrl_arr[..., None]
        ctrl_arr = jnp.broadcast_to(ctrl_arr, (*plate_shapes, *ctrl_arr.shape))
    return ctrl_arr


def _observation_noise_covariance_sequence(
    dynamics: DynamicalModel,
    *,
    obs_times: Real[Array, "... time"],
    ctrl_values: Real[Array, "... control_time control_dim"] | None,
    plate_shapes: tuple[int, ...],
) -> Float[Array, "*plate time observation_dim observation_dim"]:
    obs_times_arr = jnp.asarray(obs_times)
    t_len = int(obs_times_arr.shape[-1])
    if not _array_has_plate_dims(
        obs_times_arr,
        plate_shapes,
        min_suffix_ndim=1,
    ):
        obs_times_arr = jnp.broadcast_to(
            obs_times_arr,
            (*plate_shapes, *obs_times_arr.shape),
        )
    obs_model = dynamics.observation_model
    if isinstance(
        obs_model, (LinearGaussianObservation, GaussianObservation)
    ) and not callable(obs_model.R):
        noise_cov = jnp.asarray(obs_model.R)
        return jnp.broadcast_to(
            noise_cov[..., None, :, :],
            (*plate_shapes, t_len, *noise_cov.shape[-2:]),
        )

    state_shape = (*plate_shapes, dynamics.state_dim)
    x_probe = jnp.zeros(state_shape, dtype=obs_times_arr.dtype)

    obs_times_time_major = jnp.moveaxis(obs_times_arr, len(plate_shapes), 0)
    ctrl_values_time_major = (
        None if ctrl_values is None else jnp.moveaxis(ctrl_values, len(plate_shapes), 0)
    )

    def covariance_at_time(
        t_idx: Array,
    ) -> Float[Array, "*plate observation_dim observation_dim"]:
        t = obs_times_time_major[t_idx]
        u_t = None if ctrl_values_time_major is None else ctrl_values_time_major[t_idx]
        obs_dist = dynamics.observation_model(x_probe, u_t, t)
        if not isinstance(obs_dist, dist.MultivariateNormal):
            raise NotImplementedError(
                "Predicted observation scoring currently requires Gaussian "
                "observation models that produce MultivariateNormal distributions."
            )
        return jnp.asarray(obs_dist.covariance_matrix)

    covs_time_major = jax.lax.map(covariance_at_time, jnp.arange(t_len))
    return jnp.moveaxis(covs_time_major, 0, len(plate_shapes))


def wants_observation_prediction_diagnostics(
    filter_config: BaseFilterConfig,
) -> bool:
    """Return whether the filter should collect predictive observations."""
    return filter_config.include_predicted_observations


def _build_prediction_outputs(
    posterior: Any,
    *,
    dynamics: DynamicalModel,
    filter_config: SupportedObservationPredictionConfig,
    obs_times: Real[Array, "... time"],
    ctrl_values: Real[Array, "... control_time control_dim"]
    | Real[Array, "... control_time"]
    | None,
    plate_shapes: tuple[int, ...] = (),
) -> PredictedObservationOutputs:
    if isinstance(
        filter_config,
        (ContinuousTimeKFConfig, ContinuousTimeEKFConfig, ContinuousTimeUKFConfig),
    ):
        pred_mean_raw = getattr(posterior, "y_pred_mean", None)
        pred_cov_raw = getattr(posterior, "y_pred_cov", None)
        if pred_mean_raw is None or pred_cov_raw is None:
            raise ValueError(
                f"{type(filter_config).__name__} did not return the expected "
                "predictive observation fields."
            )
        pred_mean = _canonicalize_observations(
            jnp.asarray(pred_mean_raw),
            plate_shapes=plate_shapes,
        )
        pred_cov = jnp.asarray(pred_cov_raw)
        noise_cov = _observation_noise_covariance_sequence(
            dynamics,
            obs_times=obs_times,
            ctrl_values=_observation_control_values(
                dynamics,
                obs_times=obs_times,
                ctrl_values=ctrl_values,
                plate_shapes=plate_shapes,
            ),
            plate_shapes=plate_shapes,
        )
        obs_cov_raw = getattr(posterior, "y_obs_pred_cov", None)
        obs_cov = (
            jnp.asarray(obs_cov_raw)
            if obs_cov_raw is not None
            else pred_cov + noise_cov
        )
        return PredictedObservationOutputs(
            mean=pred_mean,
            cov=pred_cov,
            obs_cov=obs_cov,
            noise_cov=noise_cov,
        )

    if isinstance(filter_config, ContinuousTimeEnKFConfig):
        ensemble_raw = getattr(posterior, "y_ens_pred", None)
        if ensemble_raw is None:
            raise ValueError("ContinuousTimeEnKFConfig did not return `y_ens_pred`. ")
        ensemble = _canonicalize_observations(
            jnp.asarray(ensemble_raw),
            plate_shapes=plate_shapes,
        )
        pred_mean_raw = getattr(posterior, "y_pred_mean", None)
        pred_cov_raw = getattr(posterior, "y_pred_cov", None)
        if pred_mean_raw is None or pred_cov_raw is None:
            raise ValueError(
                "ContinuousTimeEnKFConfig did not return `y_pred_mean` and "
                "`y_pred_cov`."
            )
        obs_ensemble_raw = getattr(posterior, "y_obs_ens_pred", None)
        obs_ensemble = (
            _canonicalize_observations(
                jnp.asarray(obs_ensemble_raw),
                plate_shapes=plate_shapes,
            )
            if obs_ensemble_raw is not None
            else None
        )
        pred_mean = _canonicalize_observations(
            jnp.asarray(pred_mean_raw),
            plate_shapes=plate_shapes,
        )
        pred_cov = jnp.asarray(pred_cov_raw)
        noise_cov = _observation_noise_covariance_sequence(
            dynamics,
            obs_times=obs_times,
            ctrl_values=_observation_control_values(
                dynamics,
                obs_times=obs_times,
                ctrl_values=ctrl_values,
                plate_shapes=plate_shapes,
            ),
            plate_shapes=plate_shapes,
        )
        obs_cov_raw = getattr(posterior, "y_obs_pred_cov", None)
        obs_cov = (
            jnp.asarray(obs_cov_raw)
            if obs_cov_raw is not None
            else pred_cov + noise_cov
        )
        return PredictedObservationOutputs(
            mean=pred_mean,
            cov=pred_cov,
            obs_cov=obs_cov,
            ensemble=ensemble,
            obs_ensemble=obs_ensemble,
            noise_cov=noise_cov,
        )

    raise TypeError(
        f"Unsupported filter config for predicted observations: {type(filter_config).__name__}."
    )


def extract_continuous_filter_predictions(
    posterior: Any,
    *,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    obs_times: Real[Array, "... time"],
    ctrl_values: Real[Array, "... control_time control_dim"]
    | Real[Array, "... control_time"]
    | None,
    plate_shapes: tuple[int, ...] = (),
) -> PredictedObservationOutputs | None:
    """Extract canonical predicted observations when the backend supports them.

    Collection is capability-aware: unsupported filter backends keep running
    and return ``None``. An Evaluation handler decides whether missing outputs
    are an error for its requested evaluation.
    """
    if not filter_config.include_predicted_observations or not isinstance(
        filter_config,
        (
            ContinuousTimeKFConfig,
            ContinuousTimeEKFConfig,
            ContinuousTimeUKFConfig,
            ContinuousTimeEnKFConfig,
        ),
    ):
        return None

    return _build_prediction_outputs(
        posterior,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        ctrl_values=ctrl_values,
        plate_shapes=plate_shapes,
    )


def add_observation_prediction_sites(
    name: str,
    *,
    filter_config: BaseFilterConfig,
    predictions: PredictedObservationOutputs | None,
) -> None:
    """Record requested canonical predicted observations to the trace."""
    if predictions is None:
        return

    max_elems = filter_config.record_max_elems
    if predictions.mean is not None and _should_record_field(
        filter_config.record_predicted_observations_mean,
        predictions.mean.shape,
        max_elems,
    ):
        numpyro.deterministic(f"{name}_predicted_observations_mean", predictions.mean)
    if predictions.cov is not None and _should_record_field(
        filter_config.record_predicted_observations_cov,
        predictions.cov.shape,
        max_elems,
    ):
        numpyro.deterministic(f"{name}_predicted_observations_cov", predictions.cov)
    if predictions.ensemble is not None and _should_record_field(
        filter_config.record_predicted_observations_ensemble,
        predictions.ensemble.shape,
        max_elems,
    ):
        numpyro.deterministic(
            f"{name}_predicted_observations_ensemble",
            predictions.ensemble,
        )


__all__ = [
    "PredictedObservationOutputs",
    "SupportedObservationPredictionConfig",
    "add_observation_prediction_sites",
    "extract_continuous_filter_predictions",
    "wants_observation_prediction_diagnostics",
]
