"""Helpers for constructing NumPyro distributions."""

import numpyro.distributions as dist
from jax import Array
from numpyro.distributions import Distribution


def _dirac(loc: Array, event_dim: int | None = None) -> Distribution:
    """Return a scalar or vector Delta, with an optional event-axis override."""
    if event_dim is None:
        event_dim = 0 if loc.ndim == 0 else 1
    return dist.Delta(loc, event_dim=event_dim)
