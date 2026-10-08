"""Plate-axis classification for arrays, distributions, and pytree leaves."""

import jax
from jax import Array


def _array_has_plate_dims(
    arr: Array | None,
    plate_shapes: tuple[int, ...],
    *,
    min_suffix_ndim: int = 0,
) -> bool:
    """Return True when ``arr`` has ``plate_shapes`` as a leading prefix.

    ``min_suffix_ndim`` requires that many non-plate axes after the prefix, so
    callers can distinguish scalar per-member values from vector or matrix
    event values.
    """
    if arr is None:
        return False
    n_plates = len(plate_shapes)
    if arr.ndim < n_plates:
        return False
    for i, size in enumerate(plate_shapes):
        if arr.shape[i] != size:
            return False
    return (arr.ndim - n_plates) >= min_suffix_ndim


def _path_field_names(path) -> tuple[str, ...]:
    """Extract attribute names from a JAX pytree path.

    Only ``GetAttrKey`` entries (eqx ``Module`` field accesses) carry a
    meaningful ``.name`` here. ``DictKey``/``SequenceKey``/``FlattenedIndexKey``
    are intentionally dropped: built-in dynestyx model classes are eqx Modules,
    so the whitelist in ``_is_known_vector_field`` only needs attribute names.
    """
    names: list[str] = []
    for key in path:
        name = getattr(key, "name", None)
        if name is not None:
            names.append(str(name))
    return tuple(names)


# Whitelist of built-in model fields whose trailing axis is a vector event axis.
#
# A shared vector whose length happens to equal a plate size is otherwise
# ambiguous (is `(N,)` a per-member scalar or a shared length-N vector?). The
# conservative read is "shared," and `_leaf_is_plate_batched` skips rank-1
# suffixes by default. This whitelist opts specific built-in fields back in:
# for these paths, a rank-1 suffix is *known* to be a vector event axis, so
# `(N, d)` should be treated as plate-batched even when `d == 1`.
#
# Pinned by:
#   tests/test_hierarchical_smokes.py::test_unbatched_vector_fields_matching_plate_size_remain_shared
#
# To extend: add the (parent_field, ..., leaf_field) tuple here and add a
# matching smoke test exercising both the shared and plate-batched cases.
def _is_known_vector_field(path) -> bool:
    """Return True for built-in leaves whose final axis is a vector event axis."""
    names = _path_field_names(path)
    # Gaussian `Discretizer` implementations wrap the original continuous-time
    # evolution in a `cte` field, so a drift bias that lived at
    # `state_evolution.drift.b` moves to `state_evolution.cte.drift.b`. Drop the
    # private wrapper segment so the same whitelist matches discretized models.
    names = tuple(name for name in names if name != "cte")
    if len(names) >= 2 and names[-2:] in {
        ("state_evolution", "bias"),
        ("observation_model", "bias"),
    }:
        return True
    return len(names) >= 3 and names[-3:] == ("state_evolution", "drift", "b")


def _leaf_is_plate_batched(leaf, plate_shapes: tuple[int, ...], path=()) -> bool:
    """Return True if a pytree leaf should be sliced or vmapped over plates.

    Scalars with shape ``plate_shapes`` and tensors with explicit event axes are
    accepted. Rank-1 suffixes are accepted only for known vector-valued model
    fields, which protects shared vectors whose length equals a plate size.
    """
    if not isinstance(leaf, jax.Array):
        return False
    if not _array_has_plate_dims(leaf, plate_shapes, min_suffix_ndim=0):
        return False
    suffix_ndim = leaf.ndim - len(plate_shapes)
    if suffix_ndim == 1 and _is_known_vector_field(path):
        return True
    if suffix_ndim == 0 and _is_known_vector_field(path):
        return False
    return suffix_ndim == 0 or suffix_ndim >= 2


def _dist_has_plate_batch_dims(dist_obj, plate_shapes: tuple[int, ...]) -> bool:
    """Return True when a distribution's leading ``batch_shape`` matches plates."""
    if dist_obj is None or not hasattr(dist_obj, "batch_shape"):
        return False
    batch_shape = tuple(dist_obj.batch_shape)
    n_plates = len(plate_shapes)
    if len(batch_shape) < n_plates:
        return False
    for i, size in enumerate(plate_shapes):
        if batch_shape[i] != size:
            return False
    return True
