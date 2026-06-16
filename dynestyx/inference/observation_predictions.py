"""Canonical predicted-observation summaries and scoring helpers.

These utilities translate backend-specific predictive-observation fields into a
small Dynestyx-level representation used for scoring and trace recording. They
intentionally do not extend ``posterior.posterior_extras`` with Dynestyx-owned
keys, so backend storage details do not become a second public API surface.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float, Real

from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
)
from dynestyx.inference.plate_utils import _time_len_from_array
from dynestyx.inference.scoring import (
    DawidSebastianiScore,
    EnergyScore,
    GaussianLogProbScore,
    ObservationScoringConfig,
    ObservationWiseCRPSScore,
)
from dynestyx.models import DynamicalModel
from dynestyx.models.observations import GaussianObservation, LinearGaussianObservation
from dynestyx.utils import _should_record_field

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


def _compute_sample_covariance(
    ensemble: Float[Array, "*plate time n_members observation_dim"],
) -> Float[Array, "*plate time observation_dim observation_dim"]:
    n_members = ensemble.shape[-2]
    if n_members <= 1:
        raise ValueError("Predicted observation ensembles require at least 2 members.")
    mean = jnp.mean(ensemble, axis=-2)
    centered = ensemble - mean[..., None, :]
    return jnp.einsum("...tni,...tnj->...tij", centered, centered) / (n_members - 1)


def _sample_data_predictive_ensemble(
    ensemble: Float[Array, "*plate time n_members observation_dim"],
    noise_cov: Float[Array, "*plate time observation_dim observation_dim"],
    *,
    sample_seed: int,
) -> Float[Array, "*plate time n_members observation_dim"]:
    n_members = ensemble.shape[-2]
    sampled_noise = dist.MultivariateNormal(
        loc=jnp.zeros_like(ensemble[..., 0, :]),
        covariance_matrix=noise_cov,
    ).sample(jr.PRNGKey(sample_seed), sample_shape=(n_members,))
    sampled_noise = jnp.moveaxis(sampled_noise, 0, -2)
    return ensemble + sampled_noise


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
        t_len = _time_len_from_array(obs_times, plate_shapes)
        return jnp.zeros(
            (*plate_shapes, t_len, dynamics.control_dim), dtype=obs_times.dtype
        )

    ctrl_arr = jnp.asarray(ctrl_values)
    if ctrl_arr.ndim == len(plate_shapes) + 1:
        ctrl_arr = ctrl_arr[..., None]
    return ctrl_arr


def _observation_noise_covariance_sequence(
    dynamics: DynamicalModel,
    *,
    obs_times: Real[Array, "... time"],
    ctrl_values: Real[Array, "... control_time control_dim"] | None,
    plate_shapes: tuple[int, ...],
) -> Float[Array, "*plate time observation_dim observation_dim"]:
    t_len = _time_len_from_array(obs_times, plate_shapes)
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
    x_probe = jnp.zeros(state_shape, dtype=jnp.asarray(obs_times).dtype)

    obs_times_time_major = jnp.moveaxis(obs_times, len(plate_shapes), 0)
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


def _filter_requests_observation_predictions(filter_config: BaseFilterConfig) -> bool:
    return any(
        flag is True
        for flag in (
            filter_config.record_predicted_observations_mean,
            filter_config.record_predicted_observations_cov,
            filter_config.record_predicted_observations_ensemble,
        )
    )


def _filter_requests_scoring(
    scoring_config: ObservationScoringConfig | None,
) -> bool:
    return scoring_config is not None and len(scoring_config.rules) > 0


def wants_observation_prediction_diagnostics(
    filter_config: BaseFilterConfig,
    *,
    scoring_config: ObservationScoringConfig | None = None,
) -> bool:
    """Return whether predictive-observation enrichment work is needed."""
    return _filter_requests_observation_predictions(
        filter_config
    ) or _filter_requests_scoring(scoring_config)


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
    extras = posterior.posterior_extras or {}

    if isinstance(
        filter_config,
        (ContinuousTimeKFConfig, ContinuousTimeEKFConfig, ContinuousTimeUKFConfig),
    ):
        if "y_pred_mean" not in extras or "y_pred_cov" not in extras:
            raise ValueError(
                f"{type(filter_config).__name__} did not return the expected "
                "predictive observation moments in posterior_extras."
            )
        pred_mean = _canonicalize_observations(
            jnp.asarray(extras["y_pred_mean"]),
            plate_shapes=plate_shapes,
        )
        pred_cov = jnp.asarray(extras["y_pred_cov"])
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
        obs_cov = (
            jnp.asarray(extras["y_obs_pred_cov"])
            if "y_obs_pred_cov" in extras
            else pred_cov + noise_cov
        )
        return PredictedObservationOutputs(
            mean=pred_mean,
            cov=pred_cov,
            obs_cov=obs_cov,
            noise_cov=noise_cov,
        )

    if isinstance(filter_config, ContinuousTimeEnKFConfig):
        if "y_ens_pred" not in extras:
            raise ValueError(
                "ContinuousTimeEnKFConfig did not return `y_ens_pred` in "
                "posterior_extras."
            )
        ensemble = _canonicalize_observations(
            jnp.asarray(extras["y_ens_pred"]),
            plate_shapes=plate_shapes,
        )
        obs_ensemble = (
            _canonicalize_observations(
                jnp.asarray(extras["y_obs_ens_pred"]),
                plate_shapes=plate_shapes,
            )
            if "y_obs_ens_pred" in extras
            else None
        )
        pred_mean = jnp.mean(ensemble, axis=-2)
        pred_cov = _compute_sample_covariance(ensemble)
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
        obs_cov = pred_cov + noise_cov
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


def enrich_continuous_filter_output(
    posterior: Any,
    *,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    obs_times: Real[Array, "... time"],
    obs_values: Real[Array, "... time observation_dim"] | Real[Array, "... time"],
    ctrl_values: Real[Array, "... control_time control_dim"]
    | Real[Array, "... control_time"]
    | None,
    scoring_config: ObservationScoringConfig | None = None,
    plate_shapes: tuple[int, ...] = (),
) -> tuple[
    Any,
    PredictedObservationOutputs | None,
    dict[
        str,
        Float[Array, "*plate time 1"] | Float[Array, "*plate time observation_dim"],
    ],
]:
    """Compute canonical predicted-observation outputs and score arrays.

    The returned ``posterior`` is the original backend object. Canonical
    predicted-observation summaries and score arrays are returned separately
    instead of being written back into ``posterior.posterior_extras``.
    """
    wants_predictions = _filter_requests_observation_predictions(filter_config)
    wants_scores = _filter_requests_scoring(scoring_config)
    if not wants_predictions and not wants_scores:
        return posterior, None, {}

    if isinstance(filter_config, ContinuousTimeDPFConfig):
        raise NotImplementedError(
            "Predicted observation summaries and observation scoring rules are "
            "not implemented yet for ContinuousTimeDPFConfig."
        )

    if not isinstance(
        filter_config,
        (
            ContinuousTimeKFConfig,
            ContinuousTimeEKFConfig,
            ContinuousTimeUKFConfig,
            ContinuousTimeEnKFConfig,
        ),
    ):
        raise NotImplementedError(
            "Predicted observation summaries and observation scoring rules are "
            "currently supported only for continuous-time cd_dynamax Gaussian "
            "filters (KF, EKF, UKF, EnKF)."
        )

    predictions = _build_prediction_outputs(
        posterior,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        ctrl_values=ctrl_values,
        plate_shapes=plate_shapes,
    )

    score_arrays: dict[
        str,
        Float[Array, "*plate time 1"] | Float[Array, "*plate time observation_dim"],
    ] = {}
    if wants_scores:
        assert scoring_config is not None
        obs_arr = _canonicalize_observations(
            jnp.asarray(obs_values), plate_shapes=plate_shapes
        )
        score_mean, score_cov = _select_scoring_inputs(
            predictions,
        )
        for rule in scoring_config.rules:
            try:
                if isinstance(
                    rule,
                    (
                        GaussianLogProbScore,
                        DawidSebastianiScore,
                        ObservationWiseCRPSScore,
                    ),
                ):
                    score_arrays[rule.site_name] = rule.compute(
                        obs_values=obs_arr,
                        pred_mean=score_mean,
                        pred_cov=score_cov,
                    )
                elif isinstance(rule, EnergyScore):
                    score_ensemble = _select_scoring_ensemble(
                        predictions,
                        scoring_config=scoring_config,
                    )
                    score_arrays[rule.site_name] = rule.compute(
                        obs_values=obs_arr,
                        pred_mean=score_mean,
                        pred_cov=score_cov,
                        pred_ensemble=score_ensemble,
                        sample_seed=scoring_config.sample_seed,
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported observation scoring rule type: {type(rule).__name__}."
                    )
            except NotImplementedError:
                if scoring_config.unsupported == "skip":
                    continue
                raise

    return posterior, predictions, score_arrays


def _select_scoring_inputs(
    predictions: PredictedObservationOutputs,
) -> tuple[
    Float[Array, "*plate time observation_dim"],
    Float[Array, "*plate time observation_dim observation_dim"],
]:
    assert predictions.mean is not None
    if predictions.obs_cov is None:
        raise NotImplementedError(
            "Observation scoring requires predictive observation covariance."
        )
    return (
        predictions.mean,
        predictions.obs_cov,
    )


def _select_scoring_ensemble(
    predictions: PredictedObservationOutputs,
    *,
    scoring_config: ObservationScoringConfig,
) -> Float[Array, "*plate time n_members observation_dim"] | None:
    if scoring_config.sample_source == "gaussian_moments":
        return None

    if scoring_config.sample_source == "backend_ensemble":
        if predictions.obs_ensemble is not None:
            return predictions.obs_ensemble
        raise NotImplementedError(
            "Backend predictive observation ensembles are unavailable. "
            "Use `sample_source='latent_ensemble_plus_noise'` or "
            "`sample_source='gaussian_moments'`."
        )

    if scoring_config.sample_source == "latent_ensemble_plus_noise":
        if predictions.ensemble is None or predictions.noise_cov is None:
            raise NotImplementedError(
                "Synthesizing predictive observation ensembles from a latent "
                "ensemble requires both a latent predictive ensemble and "
                "observation noise covariance."
            )
        return _sample_data_predictive_ensemble(
            predictions.ensemble,
            predictions.noise_cov,
            sample_seed=scoring_config.sample_seed,
        )

    if scoring_config.sample_source == "auto":
        if predictions.obs_ensemble is not None:
            return predictions.obs_ensemble
        if predictions.ensemble is not None and predictions.noise_cov is not None:
            return _sample_data_predictive_ensemble(
                predictions.ensemble,
                predictions.noise_cov,
                sample_seed=scoring_config.sample_seed,
            )
        return None

    raise NotImplementedError(
        f"Unsupported scoring sample source: {scoring_config.sample_source}."
    )


def add_observation_prediction_and_score_sites(
    name: str,
    *,
    filter_config: BaseFilterConfig,
    scoring_config: ObservationScoringConfig | None,
    predictions: PredictedObservationOutputs | None,
    score_arrays: Mapping[
        str,
        Float[Array, "*plate time 1"] | Float[Array, "*plate time observation_dim"],
    ],
) -> None:
    """Record canonical predicted observations and score arrays to the trace."""
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

    if (
        scoring_config is None
        or not scoring_config.record_as_numpyro_sites
        or len(score_arrays) == 0
    ):
        return

    for site_name, score_arr in score_arrays.items():
        if _should_record_field(True, score_arr.shape, max_elems):
            numpyro.deterministic(f"{name}_{site_name}", score_arr)


def enrich_and_record_continuous_filter_output(
    name: str,
    posterior: Any,
    *,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    obs_times: Real[Array, "... time"],
    obs_values: Real[Array, "... time observation_dim"] | Real[Array, "... time"],
    ctrl_values: Real[Array, "... control_time control_dim"]
    | Real[Array, "... control_time"]
    | None,
    scoring_config: ObservationScoringConfig | None = None,
    plate_shapes: tuple[int, ...] = (),
) -> Any:
    """Enrich a continuous filter result and record requested trace sites."""
    posterior, predictions, score_arrays = enrich_continuous_filter_output(
        posterior,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
        plate_shapes=plate_shapes,
    )
    add_observation_prediction_and_score_sites(
        name,
        filter_config=filter_config,
        scoring_config=scoring_config,
        predictions=predictions,
        score_arrays=score_arrays,
    )
    return posterior


__all__ = [
    "PredictedObservationOutputs",
    "SupportedObservationPredictionConfig",
    "add_observation_prediction_and_score_sites",
    "enrich_continuous_filter_output",
    "enrich_and_record_continuous_filter_output",
    "wants_observation_prediction_diagnostics",
]
