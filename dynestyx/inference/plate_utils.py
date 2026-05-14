import jax
import numpyro

from dynestyx.utils import (
    _array_has_plate_dims,
    _dist_has_plate_batch_dims,
    _leaf_is_plate_batched,
    _path_field_names,
)


def _make_plate_in_axes(tree, plate_shapes: tuple[int, ...]):
    """Build ``in_axes`` for leaves whose leading dims match active plates.

    Unbatched distributions are treated as opaque shared leaves. Batched
    distributions are traversed so mixed batched/shared parameters, such as a
    batched ``MultivariateNormal.loc`` with shared covariance, can be mapped
    correctly.
    """

    def _is_unbatched_distribution_leaf(node) -> bool:
        return isinstance(
            node, numpyro.distributions.Distribution
        ) and not _dist_has_plate_batch_dims(node, plate_shapes)

    def _axis(path, leaf):
        if isinstance(leaf, numpyro.distributions.Distribution):
            return None

        if "initial_condition" in _path_field_names(path):
            return (
                0
                if _array_has_plate_dims(leaf, plate_shapes, min_suffix_ndim=1)
                else None
            )

        return 0 if _leaf_is_plate_batched(leaf, plate_shapes, path=path) else None

    return jax.tree_util.tree_map_with_path(
        _axis,
        tree,
        is_leaf=_is_unbatched_distribution_leaf,
    )


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
