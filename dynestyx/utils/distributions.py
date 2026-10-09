"""Moment extraction for Gaussian distributions."""

import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float


def gaussian_moments(
    distribution: dist.Distribution,
    event_dim: int,
) -> tuple[
    Float[Array, "*batch {event_dim}"],
    Float[Array, "*batch {event_dim} {event_dim}"],
]:
    """Return vector means and full covariances, preserving batch axes.

    Supports Normal and MultivariateNormal, including Independent and
    ExpandedDistribution wrappers. Scalar events require event_dim=1;
    vector events must have shape (event_dim,). A batch of scalar Normals
    remains a batch, rather than being interpreted as one vector event.
    """
    base = distribution
    while isinstance(base, (dist.Independent, dist.ExpandedDistribution)):
        base = base.base_dist
    if not isinstance(base, (dist.Normal, dist.MultivariateNormal)):
        raise TypeError(
            "Gaussian moments require Normal or MultivariateNormal, optionally "
            f"wrapped in Independent or ExpandedDistribution; got {type(base).__name__}."
        )
    if event_dim < 1 or distribution.event_shape not in ((), (event_dim,)):
        raise ValueError(
            f"Expected a scalar or vector Gaussian event of dimension {event_dim}; "
            f"got event_shape={distribution.event_shape}."
        )
    scalar = not distribution.event_shape
    if scalar and event_dim != 1:
        raise ValueError("A scalar Gaussian event requires event_dim=1.")

    mean = jnp.asarray(distribution.mean)
    if scalar:
        mean = mean[..., None]
    if isinstance(base, dist.MultivariateNormal):
        covariance = jnp.broadcast_to(
            base.covariance_matrix, distribution.batch_shape + (event_dim, event_dim)
        )
    else:
        variance = jnp.asarray(distribution.variance)
        if scalar:
            variance = variance[..., None]
        covariance = variance[..., :, None] * jnp.eye(event_dim, dtype=variance.dtype)
    return mean, covariance
