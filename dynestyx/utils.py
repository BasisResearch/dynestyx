import math

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from cd_dynamax import ContDiscreteNonlinearGaussianSSM as CDNLGSSM
from cd_dynamax import ContDiscreteNonlinearSSM as CDNLSSM
from jax import Array, lax

from dynestyx.models import DynamicalModel


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
    if obs_times is None:
        assert predict_times is not None
        inferred_t0 = predict_times[0]
    elif predict_times is None:
        inferred_t0 = obs_times[0]
    else:
        inferred_t0 = jnp.minimum(obs_times[0], predict_times[0])

    if dynamics.t0 is not None:
        # JIT-safe validation against user-provided t0.
        _ = eqx.error_if(
            inferred_t0,
            inferred_t0 != jnp.asarray(dynamics.t0),
            (
                f"dynamics.t0={dynamics.t0!r} does not match the earlier of obs_times[0] or predict_times[0]. "
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
    """Validate that times are strictly increasing."""
    if times is not None and len(times) > 1:
        _ = eqx.error_if(
            times,
            jnp.any(times[:-1] >= times[1:]),
            f"{name} must be strictly increasing",
        )
