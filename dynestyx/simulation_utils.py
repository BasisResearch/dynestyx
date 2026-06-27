"""Shared helpers for simulation-style trajectory outputs."""

import jax.numpy as jnp
from jax import Array


def _tile_times(times: Array, n_sim: int) -> Array:
    """Return times tiled to shape (n_sim, T)."""
    return jnp.broadcast_to(jnp.expand_dims(times, axis=0), (n_sim, len(times)))


def _ensure_trailing_dim(arr: Array) -> Array:
    """Ensure trajectory arrays follow shape (n_sim, T, dim)."""
    return arr[..., jnp.newaxis] if arr.ndim == 2 else arr
