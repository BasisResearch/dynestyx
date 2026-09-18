"""Shared normalization of independent variance and full covariance inputs."""

import jax.numpy as jnp

from dynestyx.utils import _raise_now_or_error_if


def normalize_covariance(value, layout=None):
    """Return an array and whether it represents independent variances.

    With a layout, non-scalar inputs must match its tree and event shapes.
    In particular, a matrix-shaped leaf describes pointwise variances, never
    correlations between flattened coordinates.
    """
    scalar = isinstance(value, (int, float)) or getattr(value, "ndim", None) == 0
    if layout is not None and not scalar:
        try:
            value = layout.flatten(value)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Noise must be a scalar variance or match the output Layout; "
                "full covariance matrices are not supported with a layout."
            ) from exc
    value = jnp.asarray(value)
    if not jnp.issubdtype(value.dtype, jnp.number) or jnp.iscomplexobj(value):
        raise TypeError("Gaussian covariance must be real-valued.")
    diagonal = layout is not None or value.ndim < 2
    if diagonal:
        value = _raise_now_or_error_if(
            value,
            jnp.any(~jnp.isfinite(value) | (value <= 0)),
            "Gaussian variances must be finite and strictly positive.",
        )
    elif value.shape[-2] != value.shape[-1]:
        raise ValueError("Full covariance must have square trailing axes.")
    return value, diagonal


def covariance_matrix(covariance, diagonal, dimension):
    """Expand independent variances when a backend requires dense covariance."""
    if not diagonal:
        return covariance
    covariance = jnp.asarray(covariance)
    variance = covariance + jnp.zeros((dimension,), dtype=covariance.dtype)
    return variance[..., :, None] * jnp.eye(dimension, dtype=variance.dtype)
