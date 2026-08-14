"""Evaluate predictive-observation outputs carried by conditioned results."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float

from dynestyx.evaluation.configs import ObservationScoringConfig
from dynestyx.evaluation.scoring import (
    DawidSebastianiScore,
    EnergyScore,
    GaussianLogProbScore,
    ObservationWiseCRPSScore,
)
from dynestyx.inference.observation_predictions import PredictedObservationOutputs
from dynestyx.types import ConditionedResult, EvaluationResult
from dynestyx.utils import _array_has_plate_dims


def _canonicalize_observed_values(
    arr: Float[Array, ...],
    *,
    observation_dim: int,
    plate_shapes: tuple[int, ...],
) -> Float[Array, "*plate time observation_dim"]:
    """Add a scalar event axis and broadcast observations shared by plates."""
    obs_arr = jnp.asarray(arr)
    has_plate_dims = _array_has_plate_dims(
        obs_arr,
        plate_shapes,
        min_suffix_ndim=1,
    )
    if has_plate_dims:
        if obs_arr.ndim == len(plate_shapes) + 1:
            obs_arr = obs_arr[..., None]
        return obs_arr

    if obs_arr.ndim == 1:
        obs_arr = obs_arr[..., None]
    if obs_arr.shape[-1] != observation_dim:
        raise ValueError(
            "Observation values have an incompatible trailing dimension: "
            f"expected {observation_dim}, got {obs_arr.shape[-1]}."
        )
    return jnp.broadcast_to(obs_arr, (*plate_shapes, *obs_arr.shape))


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


def _missing_prediction_error(rule_name: str, field: str, fix: str) -> str:
    return (
        f"{rule_name} requires `filtered_result.predicted_observations.{field}`, "
        "but the active filter did not provide it. "
        "Ensure predicted-observation collection is enabled with "
        "`include_predicted_observations=True` and use a filter backend that "
        f"provides this output. {fix}"
    )


def _select_scoring_ensemble(
    predictions: PredictedObservationOutputs,
    *,
    scoring_config: ObservationScoringConfig,
    rule_name: str,
) -> Float[Array, "*plate time n_members observation_dim"] | None:
    if scoring_config.sample_source == "gaussian_moments":
        return None

    if scoring_config.sample_source == "backend_ensemble":
        if predictions.obs_ensemble is not None:
            return predictions.obs_ensemble
        raise NotImplementedError(
            _missing_prediction_error(
                rule_name,
                "obs_ensemble",
                "Choose `sample_source='latent_ensemble_plus_noise'` or "
                "`sample_source='gaussian_moments'` if those inputs are available.",
            )
        )

    if scoring_config.sample_source == "latent_ensemble_plus_noise":
        if predictions.ensemble is None:
            raise NotImplementedError(
                _missing_prediction_error(rule_name, "ensemble", "")
            )
        if predictions.noise_cov is None:
            raise NotImplementedError(
                _missing_prediction_error(rule_name, "noise_cov", "")
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


def _observation_dim(predictions: PredictedObservationOutputs) -> int:
    for value in (
        predictions.mean,
        predictions.obs_ensemble,
        predictions.ensemble,
        predictions.obs_cov,
        predictions.cov,
        predictions.noise_cov,
    ):
        if value is not None:
            return int(value.shape[-1])
    raise NotImplementedError(
        "Observation scoring cannot infer the observation dimension because "
        "the active filter returned no usable predictive-observation fields. "
        "Use a filter backend that provides the predictions required by the "
        "configured rule."
    )


def compute_observation_scores(
    *,
    filtered_result: ConditionedResult,
    obs_values: Float[Array, ...] | None,
    scoring_config: ObservationScoringConfig,
    plate_shapes: tuple[int, ...] = (),
) -> dict[
    str,
    Float[Array, "*plate time 1"] | Float[Array, "*plate time observation_dim"],
]:
    """Compute configured scores from a filtered result and observed data."""
    if len(scoring_config.rules) == 0:
        return {}
    if obs_values is None:
        raise ValueError(
            "Observation scoring requires observed values. Run Evaluation around "
            "a Filter that conditioned on observations."
        )
    predictions = filtered_result.predicted_observations
    if not isinstance(predictions, PredictedObservationOutputs):
        raise ValueError(
            "Observation scoring requires "
            "`filtered_result.predicted_observations`, but the active filter did "
            "not provide canonical predictive-observation outputs. Ensure "
            "`include_predicted_observations=True` and use a supported filter "
            "backend."
        )

    obs_arr = _canonicalize_observed_values(
        obs_values,
        observation_dim=_observation_dim(predictions),
        plate_shapes=plate_shapes,
    )
    score_arrays: dict[
        str,
        Float[Array, "*plate time 1"] | Float[Array, "*plate time observation_dim"],
    ] = {}

    gaussian_rules = (
        GaussianLogProbScore,
        DawidSebastianiScore,
        ObservationWiseCRPSScore,
    )
    for rule in scoring_config.rules:
        try:
            if isinstance(rule, gaussian_rules):
                if predictions.mean is None:
                    raise NotImplementedError(
                        _missing_prediction_error(rule.site_name, "mean", "")
                    )
                if predictions.obs_cov is None:
                    raise NotImplementedError(
                        _missing_prediction_error(rule.site_name, "obs_cov", "")
                    )
                score_arrays[rule.site_name] = rule.compute(
                    obs_values=obs_arr,
                    pred_mean=predictions.mean,
                    pred_cov=predictions.obs_cov,
                )
            elif isinstance(rule, EnergyScore):
                score_ensemble = _select_scoring_ensemble(
                    predictions,
                    scoring_config=scoring_config,
                    rule_name=rule.site_name,
                )
                score_arrays[rule.site_name] = rule.compute(
                    obs_values=obs_arr,
                    pred_mean=predictions.mean,
                    pred_cov=predictions.obs_cov,
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

    return score_arrays


def build_evaluation_result(
    *,
    filtered_result: ConditionedResult,
    obs_values: Float[Array, ...] | None,
    scoring_config: ObservationScoringConfig,
    plate_shapes: tuple[int, ...] = (),
) -> EvaluationResult:
    """Build an evaluation result and its deferred NumPyro registration."""
    scores = compute_observation_scores(
        filtered_result=filtered_result,
        obs_values=obs_values,
        scoring_config=scoring_config,
        plate_shapes=plate_shapes,
    )

    def _register(site_name: str) -> None:
        if not scoring_config.record_as_numpyro_sites:
            return
        for score_name, values in scores.items():
            numpyro.deterministic(f"{site_name}_{score_name}", values)

    return EvaluationResult(
        observation_scores=scores,
        _register_numpyro_sites=_register,
    )


__all__ = [
    "build_evaluation_result",
    "compute_observation_scores",
]
