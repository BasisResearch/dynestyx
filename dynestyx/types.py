"""Shared typing helpers for dynamical systems."""

from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Real


@runtime_checkable
class FunctionOfTime(Protocol):
    def __call__(
        self, t: float | int | Real[Array, ""]
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        raise NotImplementedError()


def as_scalar_time_array(
    value: float | int | Array, *, name: str, dtype=None
) -> Real[Array, ""]:
    """Normalize a scalar time-like value to a 0-D JAX array."""
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim != 0 or jnp.issubdtype(arr.dtype, jnp.bool_):
        raise ValueError(
            f"{name} must be a numeric scalar (Python/NumPy real or scalar JAX array)."
        )
    return arr
