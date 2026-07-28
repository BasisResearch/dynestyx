"""Slice and reshape values that carry NumPyro plate dimensions."""

from contextlib import contextmanager

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Int, Shaped

from dynestyx.models import Diffusion, DynamicalModel
from dynestyx.utils import (
    _array_has_plate_dims,
    _diffusion_coefficient_is_plate_batched,
    _dist_has_plate_batch_dims,
    _is_opaque_plate_leaf,
    _leaf_is_plate_batched,
)

type PlateIndex = tuple[int | Int[Array, ""], ...]


def _make_plate_in_axes(tree, plate_shapes: tuple[int, ...]):
    """Build a `jax.vmap` axis tree for values with plate dimensions.

    Uses the following rules:
    - Distributions are always marked unbatched, and handled separately
    by `_slice_dist_for_plate_member`.
    - Diffusion coefficients are marked batched based on a test by
    `_diffusion_coefficient_is_plate_batched`.
    - All other leaves are marked as batched based on a test by
    `_leaf_is_plate_batched`.

    Each of these tests varies a bit; at a basic level, each tests for
    the presence of prepending `plate_shapes`, but have their own set
    of exceptions. Please see the docuemntation for each test
    for more details.

    Args:
        tree: Pytree whose leaves may contain plate dimensions.
        plate_shapes: Sizes of the leading plate dimensions.

    Returns:
        PyTree: Tree with the same structure as `tree` and leaves set to `0` or
            `None`.
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


def _array_plate_axis(
    arr: Shaped[Array, "..."] | None, plate_shapes: tuple[int, ...]
) -> int | None:
    return 0 if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1) else None


def _get_time_axis(plate_shapes: tuple[int, ...]) -> int:
    return len(plate_shapes)


def _time_len_from_array(
    arr: Shaped[Array, "..."], plate_shapes: tuple[int, ...]
) -> int:
    return int(arr.shape[_get_time_axis(plate_shapes)])


def _slice_time_axis(
    arr: Shaped[Array, "..."], t: int, plate_shapes: tuple[int, ...]
) -> Shaped[Array, "..."]:
    """Select one time while preserving all leading plate dimensions.

    Args:
        arr: Array shaped as `(*plate_shapes, time, ...)`.
        t: Index to select from the time axis.
        plate_shapes: Sizes of the leading plate dimensions.

    Returns:
        Array: Selected values shaped as `(*plate_shapes, ...)`.
    """
    time_axis = _get_time_axis(plate_shapes)
    return arr[(slice(None),) * time_axis + (t, ...)]


def _slice_array_for_plate_member(
    arr: Shaped[Array, "..."] | None,
    plate_shapes: tuple[int, ...],
    plate_idx: PlateIndex,
) -> Shaped[Array, "..."] | None:
    """Select one plate member from an optional array.

    An array is sliced only when it starts with `plate_shapes` and has at least
    one remaining dimension. Shared arrays and `None` are returned unchanged.
    Entries in `plate_idx` may be Python integers or scalar JAX arrays.

    Args:
        arr: Array that may contain leading plate dimensions, or `None`.
        plate_shapes: Sizes of the leading plate dimensions.
        plate_idx: One index for each plate dimension.

    Returns:
        Array | None: Array for the selected plate member, the unchanged shared
            array, or `None`.
    """
    if arr is None:
        return None
    if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1):
        return arr[plate_idx]
    return arr


def _slice_dist_for_plate_member(
    dist_obj: numpyro.distributions.Distribution,
    plate_shapes: tuple[int, ...],
    plate_idx: PlateIndex,
) -> numpyro.distributions.Distribution:
    """Return a NumPyro distribution for one plate member.

    Direct `jax.vmap` slicing can leave the original `batch_shape` in a
    distribution's static data. This can make `mean`, `sample`, and `log_prob`
    expand back to the full plate shape. This function instead rebuilds wrapper
    distributions recursively. For other distributions, it broadcasts their
    parameter leaves to the full plate shape, selects `plate_idx`, and removes
    the selected dimensions from `batch_shape`.

    A distribution whose batch shape does not start with `plate_shapes` is
    shared and returned unchanged. Entries in `plate_idx` may be Python
    integers or scalar JAX arrays.

    Args:
        dist_obj: NumPyro distribution that may contain plate batch dimensions.
        plate_shapes: Sizes of the leading plate dimensions.
        plate_idx: One index for each plate dimension.

    Returns:
        numpyro.distributions.Distribution: Distribution for the selected plate
            member, or the unchanged shared distribution.
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


def _slice_tree_for_plate_member(
    tree,
    plate_shapes: tuple[int, ...],
    plate_idx: PlateIndex,
):
    """Select one plate member from every matching leaf in a pytree.

    Member-specific array leaves are indexed by `plate_idx`. Classification
    uses both shape and the leaf's location in the tree so shared vectors are
    not sliced only because their length matches a plate size. For a
    constant-coefficient `Diffusion`, only the coefficient is sliced. Shared
    leaves are returned unchanged.

    Args:
        tree: Pytree whose leaves may contain leading plate dimensions.
        plate_shapes: Sizes of the leading plate dimensions.
        plate_idx: One index for each plate dimension.

    Returns:
        PyTree: Copy of `tree` containing values for the selected plate member.
    """

    def _slice_leaf(path, leaf):
        if isinstance(leaf, Diffusion):
            if _diffusion_coefficient_is_plate_batched(leaf, plate_shapes):
                return eqx.tree_at(
                    lambda d: d.coefficient,
                    leaf,
                    leaf.coefficient[plate_idx],
                )
            return leaf
        if _leaf_is_plate_batched(leaf, plate_shapes, path=path):
            return leaf[plate_idx]
        return leaf

    return jax.tree_util.tree_map_with_path(
        _slice_leaf,
        tree,
        is_leaf=_is_opaque_plate_leaf,
    )


def _slice_dynamics_for_plate_member(
    dynamics: DynamicalModel,
    plate_shapes: tuple[int, ...],
    plate_idx: PlateIndex,
) -> DynamicalModel:
    """Return a dynamical model for one plate member.

    The function slices matching leaves in the model and separately rebuilds
    its NumPyro initial-condition distribution. Rebuilding the distribution
    prevents its static `batch_shape` from retaining the removed plate
    dimensions.

    Args:
        dynamics: Dynamical model whose values may contain plate dimensions.
        plate_shapes: Sizes of the leading plate dimensions.
        plate_idx: One index for each plate dimension.

    Returns:
        DynamicalModel: Model containing values for the selected plate member.
    """
    member_dynamics = _slice_tree_for_plate_member(
        dynamics,
        plate_shapes,
        plate_idx,
    )
    if not _dist_has_plate_batch_dims(dynamics.initial_condition, plate_shapes):
        return member_dynamics

    member_initial_condition = _slice_dist_for_plate_member(
        dynamics.initial_condition,
        plate_shapes,
        plate_idx,
    )
    return eqx.tree_at(
        lambda model: model.initial_condition,
        member_dynamics,
        member_initial_condition,
        is_leaf=lambda value: value is None,
    )


def _stack_optional_member_values(
    values: list[Shaped[Array, "..."] | None],
    plate_shapes: tuple[int, ...],
) -> Shaped[Array, "..."] | None:
    """Stack per-member values and restore their leading plate dimensions.

    Args:
        values: One value per plate member, ordered by flattened plate index.
            Non-`None` values must have broadcast-compatible shapes.
        plate_shapes: Sizes of the plate dimensions to restore.

    Returns:
        Array | None: Stacked array shaped as
            `(*plate_shapes, *member_shape)`. Returns `None` if any member value
            is `None`.
    """
    if any(value is None for value in values):
        return None
    arrays = jnp.broadcast_arrays(*[jnp.asarray(value) for value in values])
    first = arrays[0]
    return jnp.stack(arrays).reshape(
        *plate_shapes,
        *first.shape,
    )


@contextmanager
def _suspend_numpyro_plate_frames():
    """Temporarily remove active NumPyro plate frames.

    This allows per-member sample sites to be registered without NumPyro adding
    the surrounding plate dimensions. The original effect stack is restored
    when the context exits, including when an exception is raised.
    """
    stack = numpyro.primitives._PYRO_STACK
    original = list(stack)
    stack[:] = [
        frame for frame in original if not isinstance(frame, numpyro.primitives.plate)
    ]
    try:
        yield
    finally:
        stack[:] = original
