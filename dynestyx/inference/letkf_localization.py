"""Validate explicit LETKF neighborhoods without changing their static shape."""

import jax.numpy as jnp

from dynestyx.inference.configs.filter import LETKFConfig
from dynestyx.utils.validation import _raise_now_or_error_if


def validate_letkf_localization(indices, weights, *, state_dim, observation_dim):
    """Validate a neighborhood pair eagerly or within a compiled filter."""
    indices, weights = jnp.asarray(indices), jnp.asarray(weights)
    if indices.ndim != 2 or indices.shape[0] != state_dim:
        raise ValueError(
            "LETKF observation_indices must have shape "
            f"({state_dim}, max_local_observations); got {indices.shape}."
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

    def validate(pair):
        return validate_letkf_localization(
            *pair, state_dim=state_dim, observation_dim=observation_dim
        )

    if config.get_localization is None:
        pair = validate((config.observation_indices, config.localization_weights))
        return lambda model_inputs: pair

    callback = config.get_localization
    return lambda model_inputs: validate(callback(model_inputs))
