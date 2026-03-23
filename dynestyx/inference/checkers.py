"""Validation helpers for inference modules."""

import jax
import numpyro

from dynestyx.models import DynamicalModel


def _leading_dims(arr: jax.Array | None, n_dims: int) -> tuple[int, ...] | None:
    """Return up to n_dims leading dimensions for diagnostics."""
    if arr is None:
        return None
    n = min(n_dims, arr.ndim)
    return tuple(int(d) for d in arr.shape[:n])


def _summarize_dynamics_leading_dims(
    dynamics: DynamicalModel, n_dims: int, max_items: int = 6
) -> str:
    """Summarize leading dimensions from JAX-array leaves in a model pytree."""
    shapes: list[tuple[int, ...]] = []
    for leaf in jax.tree.leaves(
        dynamics,
        is_leaf=lambda node: isinstance(node, numpyro.distributions.Distribution),
    ):
        if isinstance(leaf, jax.Array):
            n = min(n_dims, leaf.ndim)
            shapes.append(tuple(int(d) for d in leaf.shape[:n]))

    unique_shapes: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            unique_shapes.append(shape)

    if len(unique_shapes) <= max_items:
        return str(unique_shapes)
    return f"{unique_shapes[:max_items]} (+{len(unique_shapes) - max_items} more)"


def _validate_batched_plate_alignment(
    dynamics: DynamicalModel,
    plate_shapes: tuple[int, ...],
    dyn_axes,
    ot_axis: int | None,
    ov_axis: int | None,
    ct_axis: int | None,
    cv_axis: int | None,
    *,
    obs_times: jax.Array | None,
    obs_values: jax.Array | None,
    ctrl_times: jax.Array | None,
    ctrl_values: jax.Array | None,
) -> None:
    """Raise early when plate_shapes do not align with any batched input source."""
    has_batched_dynamics = any(axis == 0 for axis in jax.tree.leaves(dyn_axes))
    has_batched_data = any(axis == 0 for axis in (ot_axis, ov_axis, ct_axis, cv_axis))
    if has_batched_dynamics or has_batched_data:
        return

    n_plates = len(plate_shapes)
    diagnostics = (
        "Plate/data shape alignment failed before batched filtering. "
        f"plate_shapes={plate_shapes}. No dynamics leaves or observed/control arrays "
        "have leading dimensions matching plate_shapes. "
        f"dynamics_leading_dims={_summarize_dynamics_leading_dims(dynamics, n_plates)}; "
        f"obs_times_leading_dims={_leading_dims(obs_times, n_plates)}; "
        f"obs_values_leading_dims={_leading_dims(obs_values, n_plates)}; "
        f"ctrl_times_leading_dims={_leading_dims(ctrl_times, n_plates)}; "
        f"ctrl_values_leading_dims={_leading_dims(ctrl_values, n_plates)}. "
        "Expected at least one batched source to start with plate_shapes."
    )
    raise ValueError(diagnostics)
