"""Implements systematic resampling.

This is a copy of cuthbertlib.resampling.systematic, but uses only the default resampling function.

This is because the numba version is implemented as a jax.pure_callback, which causes issues with using gradient-based sampling methods.
"""

from functools import partial

from cuthbertlib.resampling.protocols import (
    resampling_decorator,
)
from cuthbertlib.resampling.systematic import (
    _DESCRIPTION,
)
from cuthbertlib.resampling.systematic import (
    conditional_resampling as conditional_resampling_original,
)
from cuthbertlib.resampling.systematic import (
    conditional_resampling_0_to_0 as conditional_resampling_0_to_0_original,
)
from cuthbertlib.resampling.utils import _inverse_cdf_default
from cuthbertlib.types import Array, ArrayLike
from jax import numpy as jnp
from jax import random
from jax.scipy.special import logsumexp


@partial(resampling_decorator, name="Systematic", desc=_DESCRIPTION)
def resampling(key: Array, logits: ArrayLike, n: int) -> Array:
    us = (random.uniform(key, ()) + jnp.arange(n)) / n
    weights = jnp.exp(jnp.asarray(logits) - logsumexp(logits))
    return _inverse_cdf_default(us, weights)


conditional_resampling = conditional_resampling_original
conditional_resampling_0_to_0 = conditional_resampling_0_to_0_original
