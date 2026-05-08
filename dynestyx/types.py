"""Type aliases for dynamical systems."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

State = jax.Array
dState = State
Observation = jax.Array
Control = State | None
Time = jax.Array
TimeLike = float | jax.Array
Params = dict[str, float | jax.Array]
Key = jax.Array
FunctionOfTime = Callable[[Time], State]


def as_scalar_time_array(value: TimeLike, *, name: str, dtype=None) -> Time:
    """Normalize a scalar time-like value to a 0-D JAX array.

    Args:
        value: Time-like value to normalize.
        name: Name of the time-like value for error messages.
        dtype: Optional dtype for the resulting array.

    Returns:
        Time: 0-D JAX array representing the time-like value.

    Raises:
        ValueError: If the value is not a numeric scalar (Python/NumPy real or scalar JAX array).
    """
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim != 0 or jnp.issubdtype(arr.dtype, jnp.bool_):
        raise ValueError(
            f"{name} must be a numeric scalar (Python/NumPy real or scalar JAX array)."
        )
    return arr
