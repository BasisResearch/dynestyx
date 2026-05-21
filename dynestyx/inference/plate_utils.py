import jax
import jax.numpy as jnp
import numpyro
from jaxtyping import Array, Shaped

from dynestyx.inference.integrations.utils import WeightedParticles
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
    arr: Array | None,
    plate_shapes: tuple[int, ...],
    plate_idx: tuple[int | Array, ...],
) -> Array | None:
    """Slice leading plate dims if present; otherwise return unchanged."""
    if arr is None:
        return None
    if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1):
        return arr[plate_idx]
    return arr


def _slice_tree_for_plate_member(
    tree,
    plate_shapes: tuple[int, ...],
    plate_idx: tuple[int | Array, ...],
):
    """Slice plate-batched non-distribution leaves for one plate member."""

    def _is_distribution_leaf(node) -> bool:
        return isinstance(node, numpyro.distributions.Distribution)

    def _slice_leaf(path, leaf):
        if _leaf_is_plate_batched(leaf, plate_shapes, path=path):
            return leaf[plate_idx]
        return leaf

    return jax.tree_util.tree_map_with_path(
        _slice_leaf,
        tree,
        is_leaf=_is_distribution_leaf,
    )


def _slice_dist_for_plate_member(
    dist_obj,
    plate_shapes: tuple[int, ...],
    plate_idx: tuple[int | Array, ...],
):
    """Slice plate-batched distribution parameters for one plate member."""
    if not _dist_has_plate_batch_dims(dist_obj, plate_shapes):
        return dist_obj

    def _slice_required_array(arr_like) -> Array:
        arr = jnp.asarray(arr_like)
        sliced = _slice_array_for_plate_member(arr, plate_shapes, plate_idx)
        if sliced is None:
            raise ValueError("Expected a concrete array when slicing plate member.")
        return sliced

    if isinstance(dist_obj, numpyro.distributions.MultivariateNormal):
        loc = _slice_required_array(dist_obj.loc)
        cov = _slice_required_array(dist_obj.covariance_matrix)
        return numpyro.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=cov,
        )

    if isinstance(dist_obj, numpyro.distributions.Delta):
        value = _slice_required_array(dist_obj.v)
        log_density = _slice_required_array(dist_obj.log_density)
        return numpyro.distributions.Delta(
            value,
            log_density=log_density,
            event_dim=dist_obj.event_dim,
        )

    if isinstance(dist_obj, WeightedParticles):
        particles = _slice_required_array(dist_obj.particles)
        log_weights = _slice_required_array(dist_obj.log_weights)
        return WeightedParticles(particles=particles, log_weights=log_weights)

    if dist_obj.__class__.__name__.startswith("Categorical"):
        if dist_obj.logits is not None:
            logits = _slice_required_array(dist_obj.logits)
            return numpyro.distributions.Categorical(logits=logits)
        probs = _slice_required_array(dist_obj.probs)
        return numpyro.distributions.Categorical(probs=probs)

    if isinstance(dist_obj, numpyro.distributions.Independent):
        base = _slice_dist_for_plate_member(dist_obj.base_dist, plate_shapes, plate_idx)
        return numpyro.distributions.Independent(
            base,
            dist_obj.reinterpreted_batch_ndims,
        )

    if isinstance(dist_obj, numpyro.distributions.TransformedDistribution):
        base = _slice_dist_for_plate_member(dist_obj.base_dist, plate_shapes, plate_idx)
        transforms = getattr(dist_obj, "transforms")
        return numpyro.distributions.TransformedDistribution(base, transforms)

    def _slice_leaf(leaf):
        if isinstance(leaf, jax.Array) and _array_has_plate_dims(
            leaf, plate_shapes, min_suffix_ndim=1
        ):
            return leaf[plate_idx]
        return leaf

    return jax.tree.map(_slice_leaf, dist_obj)
