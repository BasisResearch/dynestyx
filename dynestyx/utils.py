import math

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM as CDNLGSSM
from cd_dynamax import ContDiscreteNonlinearSSM as CDNLSSM
from jax import Array, lax

from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.models.checkers import _infer_bm_dim


def flatten_draws(arr: Array) -> Array:
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


type SSMType = CDNLGSSM | CDNLSSM

_CONTROL_EXTEND_EPSILON = 1e-5


def _array_has_plate_dims(
    arr: Array | None,
    plate_shapes: tuple[int, ...],
    *,
    min_suffix_ndim: int = 0,
) -> bool:
    """Return True if arr has leading dims exactly matching plate_shapes."""
    if arr is None:
        return False
    n_plates = len(plate_shapes)
    if arr.ndim < n_plates:
        return False
    for i, size in enumerate(plate_shapes):
        if arr.shape[i] != size:
            return False
    return (arr.ndim - n_plates) >= min_suffix_ndim


def _leaf_is_plate_batched(leaf, plate_shapes: tuple[int, ...]) -> bool:
    """Return True if a pytree leaf is plate-batched according to the suffix_ndim heuristic.

    A JAX array leaf is considered plate-batched when its leading dimensions
    match ``plate_shapes`` and:
    - ``suffix_ndim == 0``: per-member scalar parameter (e.g. ``beta[M]``), or
    - ``suffix_ndim >= 2``: canonical batched tensor (e.g. ``A[M, d, d]``).

    ``suffix_ndim == 1`` is intentionally skipped to avoid ambiguous false
    positives where unbatched vectors happen to start with a plate-sized
    dimension (e.g. HMM state dim == plate size).
    """
    if not isinstance(leaf, jax.Array):
        return False
    if not _array_has_plate_dims(leaf, plate_shapes, min_suffix_ndim=0):
        return False
    suffix_ndim = leaf.ndim - len(plate_shapes)
    return suffix_ndim == 0 or suffix_ndim >= 2


def _dist_has_plate_batch_dims(dist_obj, plate_shapes: tuple[int, ...]) -> bool:
    """Return True when a distribution has plate-shaped leading batch dims."""
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


def _has_any_batched_plate_source(
    dynamics: DynamicalModel,
    plate_shapes: tuple[int, ...],
    *,
    arrays: tuple[Array | None, ...] = (),
    dists: list | None = None,
) -> bool:
    """Return True if any source (dynamics leaves, arrays, or dists) is plate-batched."""
    for leaf in jax.tree.leaves(
        dynamics,
        is_leaf=lambda node: isinstance(node, numpyro.distributions.Distribution),
    ):
        if _leaf_is_plate_batched(leaf, plate_shapes):
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


def _ensure_continuous_bm_dim(dynamics: DynamicalModel) -> DynamicalModel:
    """Infer and set bm_dim when continuous dynamics were constructed in plates."""
    if not dynamics.continuous_time:
        return dynamics

    state_evolution = dynamics.state_evolution
    if (
        not isinstance(state_evolution, ContinuousTimeStateEvolution)
        or state_evolution.diffusion_coefficient is None
        or state_evolution.bm_dim is not None
    ):
        return dynamics

    x0 = jnp.zeros((dynamics.state_dim,))
    u0 = None if dynamics.control_dim == 0 else jnp.zeros((dynamics.control_dim,))
    t0 = jnp.array(0.0) if dynamics.t0 is None else jnp.asarray(dynamics.t0)
    inferred_bm_dim = _infer_bm_dim(state_evolution, dynamics.state_dim, x0, u0, t0)
    if inferred_bm_dim is not None:
        object.__setattr__(state_evolution, "bm_dim", inferred_bm_dim)
    return dynamics


def _should_record_field(
    record_val: bool | None, shape: tuple[int, ...], max_elems: int
) -> bool:
    """
    Decide whether to record a field based on user preference and size.

    - If record_val is True: always record (obey user).
    - If record_val is False: never record (obey user).
    - If record_val is None (unspecified): record only if math.prod(shape) <= max_elems.
    """
    if record_val is True:
        return True
    if record_val is False:
        return False
    return math.prod(shape) <= max_elems


def _validate_control_dim(dynamics: DynamicalModel, ctrl_values: Array | None) -> None:
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


def _validate_controls(
    obs_times: Array | None,
    predict_times: Array | None,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
) -> None:
    """
    Validate control inputs against model time grids.

    Rules:
    - ctrl_times and ctrl_values must be provided together (or both omitted).
    - At least one of obs_times or predict_times must be provided.
    - If both obs_times and predict_times are present, ctrl_times must match their union.
    - Otherwise ctrl_times must match whichever single grid is provided.
    - Matching is set-like (order-insensitive) and length-preserving.

    Raises:
        ValueError: If controls are partially provided or no time grid is provided.
    """

    if ctrl_times is None:
        if ctrl_values is not None:
            raise ValueError(
                "ctrl_values is not None, but ctrl_times is None. "
                "Provide both ctrl_times and ctrl_values together."
            )
        return
    if ctrl_values is None:
        raise ValueError(
            "ctrl_times is not None, but ctrl_values is None. "
            "Provide both ctrl_times and ctrl_values together."
        )

    if obs_times is None and predict_times is None:
        raise ValueError("At least one of obs_times or predict_times must be provided")

    if obs_times is None:
        total_obs_pred_times = predict_times
    elif predict_times is None:
        total_obs_pred_times = obs_times
    else:
        # Skip union when traced (jnp.union1d/jnp.unique fail under lax.map/vmap)
        try:
            total_obs_pred_times = jnp.union1d(obs_times, predict_times)
        except Exception:
            return  # ConcretizationTypeError etc. when arrays are traced
    assert total_obs_pred_times is not None

    # Use trace-safe check: same length and sorted arrays match.
    # (Avoid jnp.setxor1d/jnp.unique which have data-dependent output shapes and fail under JIT.)
    len_mismatch = ctrl_times.shape[0] != total_obs_pred_times.shape[0]
    values_mismatch = lax.cond(
        len_mismatch,
        lambda: jnp.array(True),
        lambda: ~jnp.allclose(jnp.sort(ctrl_times), jnp.sort(total_obs_pred_times)),
    )
    _ = eqx.error_if(
        ctrl_times,
        jnp.logical_or(len_mismatch, values_mismatch),
        "Control times and the union of obs_times and predict_times must be the same.",
    )


def _build_control_path(
    ctrl_times: Array, ctrl_values: Array, obs_times: Array
) -> dfx.LinearInterpolation:
    """
    Build rectilinear control path for continuous-time simulators.

    Extends the path past the final time so that evaluate(t_last, left=False)
    returns the last value instead of NaN (rectilinear path has no right piece
    at the boundary).
    """
    t_final = jnp.maximum(obs_times[-1], ctrl_times[-1]) + _CONTROL_EXTEND_EPSILON
    ctrl_times_ext = jnp.concatenate([ctrl_times, t_final[None]])
    ctrl_values_ext = jnp.concatenate([ctrl_values, ctrl_values[-1:]], axis=0)
    _ct, _cv = dfx.rectilinear_interpolation(ts=ctrl_times_ext, ys=ctrl_values_ext)
    return dfx.LinearInterpolation(ts=_ct, ys=_cv)


def _get_val_or_None(values: Array | None, t_idx: int) -> Array | None:
    """
    Safely get value at index t_idx, returning None if values is None.

    Args:
        values: Values array or None
        t_idx: Time index to access

    Returns:
        Value at index t_idx, or None if values is None
    """
    return values[t_idx] if values is not None else None


def _get_dynamics_with_t0(
    dynamics: DynamicalModel, obs_times: Array | None, predict_times: Array | None
) -> DynamicalModel:
    """Return dynamics with t0 filled in from obs_times[0].

    If ``dynamics.t0`` is already set, it must match the earlier of``obs_times[0]`` or ``predict_times[0]`` exactly;
    otherwise a ``ValueError`` is raised. If it is ``None``, it is filled in
    from ``obs_times[0]`` or ``predict_times[0]`` (kept as a JAX scalar so the result is jittable).
    """

    # Use the first time step along the last (time) axis, then reduce across any
    # leading batch/plate dims to a scalar t0.
    def _infer_t0_from_times(times: Array) -> Array:
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


def _validate_site_sorting(times: Array | None, name: str) -> None:
    """Validate that times are strictly increasing (along the last axis)."""
    if times is not None and times.shape[-1] > 1:
        # Use slicing on the last axis to support batched time arrays.
        t_prev = times[..., :-1]
        t_next = times[..., 1:]
        _ = eqx.error_if(
            times,
            jnp.any(t_prev >= t_next),
            f"{name} must be strictly increasing",
        )
