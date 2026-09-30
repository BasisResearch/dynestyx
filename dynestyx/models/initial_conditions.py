"""Initial-state distribution utilities."""

import jax.numpy as jnp
from numpyro.distributions import Distribution

from dynestyx.models.layout import Layouts
from dynestyx.models.utils.distribution_utils import _dirac


def DiracInitialCondition(
    loc, *, layout: Layouts | None = None, event_dim: int | None = None
) -> Distribution:
    """Return a Delta initial condition, flattening a structured state if supplied.

    By default, a scalar is one scalar event and a rank-one array is one vector
    event. Set ``event_dim=0`` for a batch of scalar events without a layout.
    """
    state_layout = None if layout is None else layout.state
    flat_loc = jnp.asarray(loc) if state_layout is None else state_layout.flatten(loc)
    return _dirac(flat_loc, event_dim)
