"""Shared normalization of independent variance and full covariance inputs."""

import jax.numpy as jnp
from numpyro import distributions as dist

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


def gaussian_distribution(loc, covariance, diagonal):
    """Build a scalar/vector Gaussian, preserving leading distribution batches."""
    loc = jnp.asarray(loc)
    width = 1 if loc.ndim == 0 else loc.shape[-1]
    if diagonal:
        if covariance.ndim and covariance.shape[-1] != width:
            raise ValueError(
                f"Expected {width} diagonal variances; got {covariance.shape}."
            )
        return dist.Normal(loc, jnp.sqrt(covariance)).to_event(
            0 if loc.ndim == 0 else 1
        )
    if covariance.shape[-2:] != (width, width):
        raise ValueError(
            f"Expected covariance shape ({width}, {width}); got {covariance.shape}."
        )
    return dist.MultivariateNormal(loc=loc, covariance_matrix=covariance)


def dense_covariance(distribution):
    """Materialize covariance only at a backend boundary requiring a matrix."""
    if isinstance(distribution, dist.MultivariateNormal):
        return distribution.covariance_matrix
    variance = jnp.atleast_1d(distribution.variance)
    return covariance_matrix(variance, True, variance.shape[-1])


def covariance_matrix(covariance, diagonal, dimension):
    """Expand independent variances when a backend requires dense covariance."""
    if not diagonal:
        return covariance
    covariance = jnp.asarray(covariance)
    variance = covariance + jnp.zeros((dimension,), dtype=covariance.dtype)
    return variance[..., :, None] * jnp.eye(dimension, dtype=variance.dtype)
