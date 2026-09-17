"""Resolution and validation of structured EnKF localization configs."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import jax.numpy as jnp
from cuthbertlib.ensemble_kalman.localization import (
    construct_tapered_chol_innovation_covariance,
    gaspari_cohn,
    gaussian,
)
from jaxtyping import Array

from dynestyx.inference.configs.filter import (
    ConstructCholInnovationCovariance,
    EnKFLocalizationConfig,
    EnKFLocalizationFunctions,
    ModifyCrossCovariance,
    ModifyPredictedObservationCovariance,
)
from dynestyx.utils import _raise_now_or_error_if


@dataclasses.dataclass(frozen=True)
class ResolvedEnKFLocalization:
    """Validated callbacks ready for Cuthbert and predictive scoring."""

    modify_cross_covariance: ModifyCrossCovariance | None = None
    construct_chol_innovation_covariance: ConstructCholInnovationCovariance | None = (
        None
    )
    modify_predicted_observation_covariance: (
        ModifyPredictedObservationCovariance | None
    ) = None
    observation_taper: Array | None = None


def _validate_array(
    value,
    *,
    expected_shape: tuple[int, ...],
    name: str,
    symmetric: bool = False,
) -> Array:
    value = jnp.asarray(value)
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {value.shape}.")
    value = _raise_now_or_error_if(
        value,
        ~jnp.all(jnp.isfinite(value)),
        f"{name} must contain only finite values.",
    )
    if symmetric:
        value = _raise_now_or_error_if(
            value,
            ~jnp.allclose(value, value.T),
            f"{name} must be symmetric.",
        )
    return value


def _validate_distances(
    value,
    *,
    expected_shape: tuple[int, ...],
    name: str,
    symmetric_zero_diagonal: bool = False,
) -> Array:
    value = _validate_array(
        value,
        expected_shape=expected_shape,
        name=name,
        symmetric=symmetric_zero_diagonal,
    )
    value = _raise_now_or_error_if(
        value,
        jnp.any(value < 0),
        f"{name} must contain only nonnegative distances.",
    )
    if symmetric_zero_diagonal:
        value = _raise_now_or_error_if(
            value,
            ~jnp.allclose(jnp.diag(value), 0),
            f"{name} must have a zero diagonal.",
        )
    return value


def apply_precomputed_observation_taper(
    predicted_observation_covariance,
    observation_taper,
    *,
    observation_dim: int,
) -> Array:
    """Apply a stored marginal taper with the same output checks as resolution."""
    predicted_observation_covariance = _validate_array(
        predicted_observation_covariance,
        expected_shape=(observation_dim, observation_dim),
        name="empirical predicted-observation covariance",
    )
    return _validate_array(
        predicted_observation_covariance * observation_taper,
        expected_shape=(observation_dim, observation_dim),
        name="localized predicted-observation covariance",
    )


def _distance_taper_fn(
    config: EnKFLocalizationConfig,
) -> Callable[[Array], Array]:
    config._validate()
    if callable(config.taper):
        return config.taper

    scale = jnp.asarray(config.taper_scale)
    scale = _raise_now_or_error_if(
        scale,
        ~jnp.isfinite(scale) | (scale <= 0),
        "EnKF localization taper_scale must be finite and strictly positive.",
    )
    covariance_fn = gaspari_cohn if config.taper == "gaspari_cohn" else gaussian
    return lambda distances: covariance_fn(distances, scale)


def _resolve_distance_localization(
    config: EnKFLocalizationConfig,
    *,
    state_dim: int,
    observation_dim: int,
) -> ResolvedEnKFLocalization:
    covariance_fn = _distance_taper_fn(config)
    cross_distances = _validate_distances(
        config.state_observation_distances,
        expected_shape=(state_dim, observation_dim),
        name="state_observation_distances",
    )
    cross_taper = _validate_array(
        covariance_fn(cross_distances),
        expected_shape=(state_dim, observation_dim),
        name="state-observation taper",
    )

    def modify_cross_covariance(cross_covariance, model_inputs):
        cross_covariance = _validate_array(
            cross_covariance,
            expected_shape=(state_dim, observation_dim),
            name="empirical state-observation cross-covariance",
        )
        return _validate_array(
            cross_covariance * cross_taper,
            expected_shape=(state_dim, observation_dim),
            name="localized state-observation cross-covariance",
        )

    if config.observation_distances is None:
        return ResolvedEnKFLocalization(
            modify_cross_covariance=modify_cross_covariance,
        )

    observation_distances = _validate_distances(
        config.observation_distances,
        expected_shape=(observation_dim, observation_dim),
        name="observation_distances",
        symmetric_zero_diagonal=True,
    )
    observation_taper = _validate_array(
        covariance_fn(observation_distances),
        expected_shape=(observation_dim, observation_dim),
        name="observation taper",
        symmetric=True,
    )
    chol_taper = jnp.linalg.cholesky(observation_taper)
    chol_taper = _raise_now_or_error_if(
        chol_taper,
        ~jnp.all(jnp.isfinite(chol_taper)),
        "The observation taper must be positive definite; its Cholesky factor "
        "contains non-finite values.",
    )

    def construct_chol_innovation_covariance(
        normalized_observation_deviations,
        chol_observation_covariance,
        model_inputs,
    ):
        return _validate_array(
            construct_tapered_chol_innovation_covariance(
                normalized_observation_deviations,
                chol_taper,
                chol_observation_covariance,
            ),
            expected_shape=(observation_dim, observation_dim),
            name="localized innovation covariance factor",
        )

    def modify_predicted_observation_covariance(
        predicted_observation_covariance,
        model_inputs,
    ):
        return apply_precomputed_observation_taper(
            predicted_observation_covariance,
            observation_taper,
            observation_dim=observation_dim,
        )

    return ResolvedEnKFLocalization(
        modify_cross_covariance=modify_cross_covariance,
        construct_chol_innovation_covariance=(construct_chol_innovation_covariance),
        modify_predicted_observation_covariance=(
            modify_predicted_observation_covariance
        ),
        observation_taper=observation_taper,
    )


def _checked_callback(
    callback: Callable[..., Array] | None,
    *,
    expected_shape: tuple[int, ...],
    name: str,
) -> Callable[..., Array] | None:
    if callback is None:
        return None

    def checked(*args, **kwargs):
        return _validate_array(
            callback(*args, **kwargs), expected_shape=expected_shape, name=name
        )

    return checked


def _resolve_callback_localization(
    config: EnKFLocalizationFunctions,
    *,
    state_dim: int,
    observation_dim: int,
) -> ResolvedEnKFLocalization:
    config._validate()
    return ResolvedEnKFLocalization(
        modify_cross_covariance=_checked_callback(
            config.modify_cross_covariance,
            expected_shape=(state_dim, observation_dim),
            name="modify_cross_covariance output",
        ),
        construct_chol_innovation_covariance=_checked_callback(
            config.construct_chol_innovation_covariance,
            expected_shape=(observation_dim, observation_dim),
            name="construct_chol_innovation_covariance output",
        ),
        modify_predicted_observation_covariance=_checked_callback(
            config.modify_predicted_observation_covariance,
            expected_shape=(observation_dim, observation_dim),
            name="modify_predicted_observation_covariance output",
        ),
    )


def resolve_enkf_localization(
    localization: EnKFLocalizationConfig | EnKFLocalizationFunctions,
    *,
    state_dim: int,
    observation_dim: int,
) -> ResolvedEnKFLocalization:
    """Validate and resolve a public localization config into callback functions."""
    if isinstance(localization, EnKFLocalizationConfig):
        return _resolve_distance_localization(
            localization,
            state_dim=state_dim,
            observation_dim=observation_dim,
        )
    if isinstance(localization, EnKFLocalizationFunctions):
        return _resolve_callback_localization(
            localization,
            state_dim=state_dim,
            observation_dim=observation_dim,
        )
    raise TypeError(
        "localization must be EnKFLocalizationConfig or EnKFLocalizationFunctions."
    )


__all__ = [
    "ResolvedEnKFLocalization",
    "apply_precomputed_observation_taper",
    "resolve_enkf_localization",
]
