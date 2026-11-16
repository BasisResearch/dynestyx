"""Array conversion, event-axis normalization, and indexing helpers."""

from jax import Array
from jaxtyping import Real, Shaped


def flatten_draws(arr: Shaped[Array, "..."]) -> Shaped[Array, "..."]:
    """Merge the leading ``(num_samples, n_sim)`` axes of a simulator output into one.

    Simulators return arrays of shape ``(n_sim, T, ...)``. After wrapping the
    model in :class:`~numpyro.infer.Predictive` with ``num_samples=N``, the
    output becomes ``(N, n_sim, T, ...)``.  This helper collapses both draw axes
    so that all ``N * n_sim`` trajectories can be treated uniformly — useful for
    computing credible intervals or plotting fans.

    Args:
        arr: Array of shape ``(num_samples, n_sim, ...)``.

    Returns:
        Array of shape ``(num_samples * n_sim, ...)``.

    Example:
        >>> states = samples["f_states"]      # (num_samples, n_sim, T, state_dim)
        >>> draws = flatten_draws(states)      # (num_samples * n_sim, T, state_dim)
        >>> lo, hi = jnp.percentile(draws, jnp.array([5.0, 95.0]), axis=0)
    """
    return arr.reshape((-1,) + arr.shape[2:])


def _ensure_trailing_event_axis(
    values: Real[Array, "..."],
) -> Real[Array, "..."]:
    """Lift a scalar time series from ``(time,)`` to ``(time, 1)``."""
    if values.ndim == 1:
        return values[..., None]
    return values


def _get_val_or_None(values: Array | None, t_idx: int | Array) -> Array | None:
    """
    Safely get value at index t_idx, returning None if values is None.

    Args:
        values: Values array or None
        t_idx: Time index to access

    Returns:
        Value at index t_idx, or None if values is None
    """
    return values[t_idx] if values is not None else None
