import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Shaped

from dynestyx.models import Diffusion
from dynestyx.utils import (
    _array_has_plate_dims,
    _diffusion_coefficient_is_plate_batched,
    _dist_has_plate_batch_dims,
    _is_opaque_plate_leaf,
    _leaf_is_plate_batched,
)


def _make_plate_in_axes(tree, plate_shapes: tuple[int, ...]):
    """Build ``in_axes`` for leaves whose leading dims match active plates.

    All numpyro distributions are treated as opaque leaves with ``in_axes=None``.
    A plate-batched ``initial_condition`` is *not* sliced by ``vmap`` here; the
    batched filter/smoother dispatch rebuilds it per member from the clean
    original via ``_slice_dist_for_plate_member``. This avoids ``vmap`` leaving a
    stale ``batch_shape`` in the distribution's static aux-data (which would make
    ``.mean`` / ``.sample`` / ``.log_prob`` re-expand to the full plate shape).
    """

    def _axis(path, leaf):
        if isinstance(leaf, numpyro.distributions.Distribution):
            return None
        # Only constant-coefficient diffusions are opaque leaves (see
        # ``_is_opaque_plate_leaf``); a callable coefficient is recursed into, so
        # its array fields are vmapped generically by the branch below.
        if isinstance(leaf, Diffusion):
            return (
                0
                if _diffusion_coefficient_is_plate_batched(leaf, plate_shapes)
                else None
            )
        return 0 if _leaf_is_plate_batched(leaf, plate_shapes, path=path) else None

    return jax.tree_util.tree_map_with_path(
        _axis,
        tree,
        is_leaf=_is_opaque_plate_leaf,
    )


def _array_plate_axis(arr, plate_shapes: tuple[int, ...]):
    """Return 0 if arr has leading dims matching plate_shapes, else None."""
    return 0 if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1) else None


def _get_time_axis(plate_shapes: tuple[int, ...]) -> int:
    """Return the axis index corresponding to time after plate dimensions."""
    return len(plate_shapes)


def _time_len_from_array(
    arr: Shaped[Array, "..."], plate_shapes: tuple[int, ...]
) -> int:
    """Infer sequence length from an array with plate dims followed by time."""
    return int(arr.shape[_get_time_axis(plate_shapes)])


def _slice_time_axis(
    arr: Shaped[Array, "..."], t: int, plate_shapes: tuple[int, ...]
) -> Shaped[Array, "..."]:
    """Slice an array at time index t where time axis follows plate dims."""
    time_axis = _get_time_axis(plate_shapes)
    return arr[(slice(None),) * time_axis + (t, ...)]


def _slice_array_for_plate_member(
    arr: Array | None, plate_shapes: tuple[int, ...], plate_idx: tuple
) -> Array | None:
    """Slice leading plate dims if present; otherwise return unchanged.

    Used both by the simulator enumerate loops and by the filter/smoother batched
    dispatch to select the times/values/parameters for a particular plate member.

    ``plate_idx`` entries may be Python ints (simulator enumerate path) or traced
    scalar arrays (filter/smoother ``vmap`` path), so it is intentionally untyped.
    """
    if arr is None:
        return None
    if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1):
        return arr[plate_idx]
    return arr


def _slice_dist_for_plate_member(
    dist_obj, plate_shapes: tuple[int, ...], plate_idx: tuple
):
    """Return the single-member distribution for plate index ``plate_idx``.

    Slicing a distribution leaf with ``jax.vmap`` leaves a stale ``batch_shape``
    in NumPyro's static aux-data, so derived quantities (``.mean`` / ``.sample`` /
    ``.log_prob``) re-expand to the full plate shape. To avoid that we rebuild the
    member distribution from the clean original:

    - **Structural** distributions (``MixtureSameFamily``, ``Independent``,
      ``TransformedDistribution``) wrap sub-distributions whose own ``batch_shape``
      would otherwise stay stale, so we recurse and rebuild via their constructors.
    - **Flat** distributions are rebatched generically: broadcast each leaf's
      leading ``n = len(plate_shapes)`` dims to the real plate sizes, slice out
      ``plate_idx``, then trim the now-removed plate dims from ``batch_shape``.
      Broadcasting first means even leaves NumPyro stored with a collapsed singleton
      batch (e.g. an MVN ``scale_tril`` of shape ``(1, d, d)``) slice to a clean
      per-member shape, so no stray ``batch_shape (1,)`` survives.

    ``plate_idx`` entries may be Python ints (simulator) or traced scalar arrays
    (filter/smoother ``vmap`` path); both index correctly via gather.
    """
    if not _dist_has_plate_batch_dims(dist_obj, plate_shapes):
        return dist_obj

    if isinstance(dist_obj, numpyro.distributions.MixtureSameFamily):
        return numpyro.distributions.MixtureSameFamily(
            _slice_dist_for_plate_member(
                dist_obj.mixing_distribution, plate_shapes, plate_idx
            ),
            _slice_dist_for_plate_member(
                dist_obj.component_distribution, plate_shapes, plate_idx
            ),
        )
    if isinstance(dist_obj, numpyro.distributions.Independent):
        return numpyro.distributions.Independent(
            _slice_dist_for_plate_member(dist_obj.base_dist, plate_shapes, plate_idx),
            dist_obj.reinterpreted_batch_ndims,
        )
    if isinstance(dist_obj, numpyro.distributions.TransformedDistribution):
        return numpyro.distributions.TransformedDistribution(
            _slice_dist_for_plate_member(dist_obj.base_dist, plate_shapes, plate_idx),
            dist_obj.transforms,
        )

    n = len(plate_shapes)
    batch_shape = tuple(dist_obj.batch_shape)
    leaves, treedef = jax.tree_util.tree_flatten(dist_obj)
    sliced_leaves = []
    for leaf in leaves:
        arr = jnp.asarray(leaf)
        full = jnp.broadcast_to(arr, tuple(plate_shapes) + arr.shape[n:])
        sliced_leaves.append(full[plate_idx])
    member = jax.tree_util.tree_unflatten(treedef, sliced_leaves)
    # _batch_shape is NumPyro's static aux field (see pytree_aux_fields); trim the
    # plate dims we just sliced away so the per-member distribution is unbatched.
    object.__setattr__(member, "_batch_shape", batch_shape[n:])
    return member
