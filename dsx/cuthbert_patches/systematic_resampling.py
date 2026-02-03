"""Implements systematic resampling.

This is a copy of cuthbertlib.resampling.systematic, but uses only the default resampling function.

This is because the numba version is implemented as a jax.pure_callback, which causes issues with using gradient-based sampling methods.
"""

from functools import partial

from jax import numpy as jnp
from jax import random

from cuthbertlib.resampling.protocols import (
    resampling_decorator,
)
from cuthbertlib.types import Array, ArrayLike
from cuthbertlib.resampling.systematic import (
    _DESCRIPTION,
    conditional_resampling as conditional_resampling_original,
    conditional_resampling_0_to_0 as conditional_resampling_0_to_0_original,
)
from cuthbertlib.resampling.utils import _inverse_cdf_default


@partial(resampling_decorator, name="Systematic", desc=_DESCRIPTION)
def resampling(key: Array, logits: ArrayLike, n: int) -> Array:
    us = (random.uniform(key, ()) + jnp.arange(n)) / n
    return _inverse_cdf_default(us, logits)


conditional_resampling = conditional_resampling_original
conditional_resampling_0_to_0 = conditional_resampling_0_to_0_original
