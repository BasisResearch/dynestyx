"""Implements multinomial resampling.

This is a copy of cuthbertlib.resampling.multinomial, but uses only the default resampling function.

This is because the numba version is implemented as a jax.pure_callback, which causes issues with using gradient-based sampling methods.
"""

from functools import partial

from cuthbertlib.resampling.multinomial import (
    _DESCRIPTION,
    _sorted_uniforms,
)
from cuthbertlib.resampling.multinomial import (
    conditional_resampling as conditional_resampling_original,
)
from cuthbertlib.resampling.protocols import (
    resampling_decorator,
)
from cuthbertlib.resampling.utils import _inverse_cdf_default
from cuthbertlib.types import Array, ArrayLike
from jax import numpy as jnp
from jax import random
from jax.scipy.special import logsumexp


@partial(resampling_decorator, name="Multinomial", desc=_DESCRIPTION)
def resampling(key: Array, logits: ArrayLike, n: int) -> Array:
    # In practice we don't have to sort the generated uniforms, but searchsorted
    # works faster and is more stable if both inputs are sorted, so we use the
    # _sorted_uniforms from N. Chopin, but still use searchsorted instead of his
    # O(N) loop as our code is meant to work on GPU where searchsorted is
    # O(log(N)) anyway.
    # We then permute the indices to enforce exchangeability.

    key_uniforms, key_shuffle = random.split(key)
    sorted_uniforms = _sorted_uniforms(key_uniforms, n)

    # In cuthbert, this may use a jax.pure_callback,
    # which is incompatible with gradient-based sampling methods.
    # Here, we use the default resampling function, which is a pure jax function.
    weights = jnp.exp(jnp.asarray(logits) - logsumexp(logits))
    idx = _inverse_cdf_default(sorted_uniforms, weights)
    return random.permutation(key_shuffle, idx)


conditional_resampling = conditional_resampling_original
