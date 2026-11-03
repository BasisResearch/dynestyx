"""Validate explicit LETKF neighborhoods without changing their static shape."""

import jax.numpy as jnp

from dynestyx.inference.configs.filter import LETKFConfig
from dynestyx.utils.validation import _raise_now_or_error_if


def validate_letkf_localization(indices, weights, *, state_dim, observation_dim):
    """Validate a neighborhood pair eagerly or within a compiled filter."""
    indices, weights = jnp.asarray(indices), jnp.asarray(weights)
    if indices.ndim != 2 or (
        indices.shape[0] != state_dim
        if state_dim is not None
        else indices.shape[0] == 0
    ):
        expected_rows = state_dim if state_dim is not None else "n_analysis >= 1"
        raise ValueError(
            "LETKF observation_indices must have shape "
            f"({expected_rows}, max_local_observations); got {indices.shape}."
        )
    if weights.shape != indices.shape:
        raise ValueError(
            "LETKF localization_weights must match observation_indices shape."
        )
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError("LETKF observation_indices must contain integers.")
    if not (
        jnp.issubdtype(weights.dtype, jnp.floating)
        or jnp.issubdtype(weights.dtype, jnp.integer)
    ):
        raise ValueError("LETKF localization_weights must be real numeric values.")
    indices = _raise_now_or_error_if(
        indices,
        jnp.any((indices < -1) | (indices >= observation_dim)),
        "LETKF observation_indices must be -1 or in observation range.",
    )
    ordered = jnp.sort(indices, axis=1)
    indices = _raise_now_or_error_if(
        indices,
        jnp.any((ordered[:, 1:] == ordered[:, :-1]) & (ordered[:, 1:] >= 0)),
        "LETKF nonpadding observation_indices must be unique within each row.",
    )
    weights = _raise_now_or_error_if(
        weights,
        jnp.any((indices >= 0) & (~jnp.isfinite(weights) | (weights < 0))),
        "LETKF nonpadding localization_weights must be finite and nonnegative.",
    )
    # Padding weights are ignored, including nonfinite padding values.
    return indices, jnp.where(indices >= 0, weights, 0)


def resolve_letkf_localization(config: LETKFConfig, *, state_dim, observation_dim):
    """Resolve constant neighborhoods once or validate a per-step callback."""

    interpolate = (
        config.weight_interpolation is not None
        or config.get_weight_interpolation is not None
    )

    def validate(pair):
        return validate_letkf_localization(
            *pair,
            state_dim=None if interpolate else state_dim,
            observation_dim=observation_dim,
        )

    if config.get_localization is None:
        pair = validate((config.observation_indices, config.localization_weights))
        return lambda model_inputs: pair

    callback = config.get_localization
    return lambda model_inputs: validate(callback(model_inputs))


def validate_letkf_weight_interpolation(pair, *, state_dim, n_analysis):
    """Check interpolation geometry and retain checks under JIT."""
    indices, coefficients = (jnp.asarray(value) for value in pair)
    if (
        n_analysis < 1
        or indices.ndim != 2
        or indices.shape[0] != state_dim
        or coefficients.shape != indices.shape
    ):
        raise ValueError(
            "LETKF weight_interpolation requires at least one analysis point "
            "and matching arrays shaped (state_dim, stencil_size)."
        )
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError("LETKF interpolation indices must contain integers.")
    if not (
        jnp.issubdtype(coefficients.dtype, jnp.floating)
        or jnp.issubdtype(coefficients.dtype, jnp.integer)
    ):
        raise ValueError(
            "LETKF interpolation coefficients must be real numeric values."
        )
    indices = _raise_now_or_error_if(
        indices,
        jnp.any((indices < -1) | (indices >= n_analysis)),
        "LETKF interpolation indices must be -1 or in analysis-point range.",
    )
    coefficients = jnp.where(indices >= 0, coefficients, 0)
    coefficients = _raise_now_or_error_if(
        coefficients,
        jnp.any(~jnp.isfinite(coefficients) | (coefficients < 0)),
        "LETKF interpolation coefficients must be finite and nonnegative.",
    )
    coefficients = _raise_now_or_error_if(
        coefficients,
        ~jnp.all(jnp.isclose(jnp.sum(coefficients, axis=1), 1.0)),
        "LETKF interpolation coefficients must sum to one in every row.",
    )
    return indices, coefficients


def resolve_letkf_weight_interpolation(config, *, state_dim, get_localization):
    """Resolve static interpolation once, or validate a per-step callback."""
    if config.weight_interpolation is None and config.get_weight_interpolation is None:
        return None

    def validate(pair, n_analysis):
        return validate_letkf_weight_interpolation(
            pair, state_dim=state_dim, n_analysis=n_analysis
        )

    if config.get_localization is None and config.get_weight_interpolation is None:
        pair = validate(
            config.weight_interpolation,
            jnp.asarray(config.observation_indices).shape[0],
        )
        return lambda model_inputs: pair

    def get_interpolation(model_inputs):
        pair = (
            config.weight_interpolation
            if config.get_weight_interpolation is None
            else config.get_weight_interpolation(model_inputs)
        )
        return validate(pair, get_localization(model_inputs)[0].shape[0])

    return get_interpolation
