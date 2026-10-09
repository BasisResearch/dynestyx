"""Model-dependent utilities for plate classification and model metadata."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from jax import Array
from jaxtyping import Real

from dynestyx.models.core import DynamicalModel
from dynestyx.models.diffusions import Diffusion
from dynestyx.utils.plates import (
    _array_has_plate_dims,
    _dist_has_plate_batch_dims,
    _leaf_is_plate_batched,
)


def _diffusion_coefficient_is_plate_batched(
    diffusion: Diffusion, plate_shapes: tuple[int, ...]
) -> bool:
    """Return True if a diffusion's *constant* coefficient is laid out per-member.

    True means the coefficient is a constant array whose leading axes are exactly
    ``plate_shapes`` followed by its intrinsic event axes — i.e. it should be
    sliced (``coefficient[plate_idx]``) or vmapped (``in_axes=0``) as an opaque
    unit. Classification uses the coefficient's event rank (a static property of
    the ``Diffusion``) rather than a raw suffix-rank heuristic, so a *shared*
    matrix/vector coefficient whose shape happens to coincide with the plate sizes
    (e.g. a ``(state_dim, bm_dim)`` matrix under nested plates ``(state_dim,
    bm_dim)``) is correctly treated as shared.

    Returns False for a callable coefficient: such a coefficient is not classified
    as a unit here. The plate consumers instead recurse into a callable
    ``eqx.Module`` coefficient and slice/vmap its per-member array fields
    generically (a plain closure that captures per-member parameters remains the
    unsupported sharp edge; see the ``dsx.plate`` docstring).
    """
    event_rank = diffusion.coefficient_event_rank
    if event_rank is None:
        return False
    shape = diffusion._constant_shape()
    assert shape is not None
    n_plates = len(plate_shapes)
    return len(shape) == n_plates + event_rank and tuple(shape[:n_plates]) == tuple(
        plate_shapes
    )


def _is_opaque_plate_leaf(node) -> bool:
    """Shared ``is_leaf`` predicate for plate classification, slicing, and vmap.

    A ``Diffusion`` with a *constant* coefficient is an opaque unit (classified by
    its own ``coefficient_event_rank``, since a path-blind shape check cannot
    disambiguate it). A *callable* coefficient may be an ``eqx.Module`` carrying
    per-member array fields, so the tree must recurse into it and handle those
    fields generically. NumPyro distributions are always opaque. The three
    consumers (:func:`_has_any_batched_plate_source`,
    ``inference.utils.plate_utils._make_plate_in_axes``,
    ``simulation._slice_tree_for_plate_member``) must share this one
    predicate so
    a callable diffusion is never seen as batched by the slicer/vmap while being
    invisible to the alignment guard.
    """
    if isinstance(node, Diffusion):
        return not callable(node.coefficient)
    return isinstance(node, numpyro.distributions.Distribution)


def _has_any_batched_plate_source(
    dynamics: DynamicalModel,
    plate_shapes: tuple[int, ...],
    *,
    arrays: tuple[Array | None, ...] = (),
    dists: list | None = None,
) -> bool:
    """Return True if dynamics, arrays, or distributions carry plate axes."""
    for path, leaf in jax.tree_util.tree_flatten_with_path(
        dynamics,
        is_leaf=_is_opaque_plate_leaf,
    )[0]:
        if isinstance(leaf, numpyro.distributions.Distribution):
            if _dist_has_plate_batch_dims(leaf, plate_shapes):
                return True
            continue
        # Only constant-coefficient diffusions are opaque leaves here; a callable
        # coefficient is recursed into, so its per-member array fields reach the
        # generic ``_leaf_is_plate_batched`` branch below.
        if isinstance(leaf, Diffusion):
            if _diffusion_coefficient_is_plate_batched(leaf, plate_shapes):
                return True
            continue
        if _leaf_is_plate_batched(leaf, plate_shapes, path=path):
            return True

    if any(
        _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1) for arr in arrays
    ):
        return True

    if dists is not None and any(
        _dist_has_plate_batch_dims(dist_obj, plate_shapes) for dist_obj in dists
    ):
        return True

    return False


def _validate_control_dim(
    dynamics: DynamicalModel,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None,
) -> None:
    """
    Validate that control_dim is set in DynamicalModel when controls are present.

    Args:
        dynamics: DynamicalModel instance
        ctrl_values: Control values array or None

    Raises:
        ValueError: If controls are provided but control_dim is not set or is 0
    """
    if ctrl_values is not None:
        if dynamics.control_dim is None or dynamics.control_dim == 0:
            # Try to infer from shape
            if ctrl_values.ndim >= 2:
                inferred_dim = ctrl_values.shape[1]
                raise ValueError(
                    f"Controls are provided (shape: {ctrl_values.shape}), but "
                    f"dynamics.control_dim is {dynamics.control_dim}. "
                    f"Please set control_dim={inferred_dim} when creating the DynamicalModel."
                )
            else:
                raise ValueError(
                    f"Controls are provided, but dynamics.control_dim is {dynamics.control_dim}. "
                    "Please set control_dim when creating the DynamicalModel."
                )


def _get_dynamics_with_t0(
    dynamics: DynamicalModel,
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None,
    predict_times: Real[Array, "*predict_time_plate predict_time"] | None,
) -> DynamicalModel:
    """Return dynamics with t0 filled in from obs_times[0].

    If ``dynamics.t0`` is already set, it must match the earlier of``obs_times[0]`` or ``predict_times[0]`` exactly;
    otherwise a ``ValueError`` is raised. If it is ``None``, it is filled in
    from ``obs_times[0]`` or ``predict_times[0]`` (kept as a JAX scalar so the result is jittable).
    """

    # Use the first time step along the last (time) axis, then reduce across any
    # leading batch/plate dims to a scalar t0.
    def _infer_t0_from_times(
        times: Real[Array, "*time_plate time"],
    ) -> Real[Array, ""]:
        return jnp.min(times[..., 0])

    if obs_times is None:
        assert predict_times is not None
        inferred_t0 = _infer_t0_from_times(predict_times)
    elif predict_times is None:
        inferred_t0 = _infer_t0_from_times(obs_times)
    else:
        inferred_t0 = jnp.minimum(
            _infer_t0_from_times(obs_times),
            _infer_t0_from_times(predict_times),
        )

    if dynamics.t0 is not None:
        t0_display = dynamics.t0
        if isinstance(t0_display, Array) and t0_display.ndim == 0:
            t0_display = t0_display.item()
        # JIT-safe validation against user-provided t0.
        _ = eqx.error_if(
            inferred_t0,
            inferred_t0 != jnp.asarray(dynamics.t0),
            (
                f"dynamics.t0={t0_display!r} does not match the earlier of obs_times[0] or predict_times[0]. "
                "Either set t0=None to auto-infer from provided times, or ensure they agree."
            ),
        )
        # Return dynamics with original t0
        return dynamics
    else:
        # Return dynamics with auto-inferred t0
        return eqx.tree_at(
            lambda m: m.t0, dynamics, inferred_t0, is_leaf=lambda x: x is None
        )
