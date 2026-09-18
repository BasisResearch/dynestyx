"""Initial distributions with optional structured-state flattening."""

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import Distribution

from dynestyx.distributions._gaussian import covariance_matrix, normalize_covariance
from dynestyx.models.layout import Layout


def DiracInitialCondition(value, *, state_layout: Layout | None = None) -> Distribution:
    """Return an exact initial distribution, flattening with the supplied layout.

    Without a layout, values are scalar or vector events, with any leading
    vector batch axes preserved.
    """
    loc = jnp.asarray(value) if state_layout is None else state_layout.flatten(value)
    return dist.Delta(loc, event_dim=0 if loc.ndim == 0 else 1)


def GaussianInitialCondition(
    mean, cov, *, state_layout: Layout | None = None
) -> Distribution:
    """Return a Gaussian initial distribution in flat coordinates.

    With a layout, ``mean`` matches that layout and ``cov`` is a scalar
    variance or a matching pytree of pointwise variances. Without a layout,
    scalar variances, diagonal variance vectors, and full covariances are
    accepted. Leading batch axes on the mean are preserved. Variances are
    expanded to dense covariance; scalar means become one-coordinate vectors.
    """
    loc = jnp.asarray(mean) if state_layout is None else state_layout.flatten(mean)
    covariance, diagonal = normalize_covariance(cov, state_layout)
    loc = jnp.atleast_1d(loc)
    width = loc.shape[-1]
    if diagonal and covariance.ndim and covariance.shape[-1] != width:
        raise ValueError(
            f"Expected {width} diagonal variances; got {covariance.shape}."
        )
    covariance = covariance_matrix(covariance, diagonal, width)
    if covariance.shape[-2:] != (width, width):
        raise ValueError(
            f"Expected covariance shape ({width}, {width}); got {covariance.shape}."
        )
    return dist.MultivariateNormal(loc=loc, covariance_matrix=covariance)
