import jax
import numpyro

from dynestyx.utils import _array_has_plate_dims, _leaf_is_plate_batched


def _make_plate_in_axes(tree, plate_shapes: tuple[int, ...]):
    """Build vmap in_axes for leaves whose leading dims match active plates."""

    def _is_distribution_leaf(node) -> bool:
        return isinstance(node, numpyro.distributions.Distribution)

    def _axis(leaf):
        return 0 if _leaf_is_plate_batched(leaf, plate_shapes) else None

    return jax.tree.map(_axis, tree, is_leaf=_is_distribution_leaf)


def _array_plate_axis(arr, plate_shapes: tuple[int, ...]):
    """Return 0 if arr has leading dims matching plate_shapes, else None."""
    return 0 if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1) else None


def _get_time_axis(plate_shapes: tuple[int, ...]) -> int:
    """Return the axis index corresponding to time after plate dimensions."""
    return len(plate_shapes)


def _time_len_from_array(arr: jax.Array, plate_shapes: tuple[int, ...]) -> int:
    """Infer sequence length from an array with plate dims followed by time."""
    return int(arr.shape[_get_time_axis(plate_shapes)])


def _slice_time_axis(
    arr: jax.Array, t: int, plate_shapes: tuple[int, ...]
) -> jax.Array:
    """Slice an array at time index t where time axis follows plate dims."""
    time_axis = _get_time_axis(plate_shapes)
    return arr[(slice(None),) * time_axis + (t, ...)]
