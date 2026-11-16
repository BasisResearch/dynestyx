"""Control time-grid validation and continuous-time path evaluation."""

from collections.abc import Callable

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import Real

_CONTROL_EXTEND_EPSILON = 1e-5


def _validate_controls(
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None,
    predict_times: Real[Array, "*predict_time_plate predict_time"] | None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None,
    *,
    observation_control_alignment: str | None = None,
) -> None:
    """
    Validate control inputs against model time grids.

    Rules:
    - ctrl_times and ctrl_values must be provided together (or both omitted).
    - At least one of obs_times or predict_times must be provided.
    - If both obs_times and predict_times are present, ctrl_times must match their union.
    - Otherwise ctrl_times must match whichever single grid is provided.
    - Matching is set-like (order-insensitive) and length-preserving.
    - When observation_control_alignment is "previous_transition", ctrl_times must
      instead match predict_times[:-1] (one control per transition); obs_times-based
      conditioning is not supported yet under this convention (see issue #312).

    Raises:
        ValueError: If controls are partially provided or no time grid is provided.
    """

    if observation_control_alignment == "previous_transition" and obs_times is not None:
        raise ValueError(
            "observation_control_alignment='previous_transition' does not "
            "support obs_times-based conditioning yet (Filter/Smoother/"
            "LatentPathBuilder posterior rollout); only predict_times-only "
            "generation is supported. See issue #312."
        )

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

    if observation_control_alignment == "previous_transition":
        # obs_times is None here -- already rejected above otherwise.
        assert predict_times is not None
        total_obs_pred_times = predict_times[..., :-1]
    elif obs_times is None:
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

    # Check that the number of control times matches the number of observation/prediction times.
    ctrl_time_count = ctrl_times.shape[-1]
    expected_time_count = total_obs_pred_times.shape[-1]
    if ctrl_time_count != expected_time_count:
        raise ValueError(
            "Control times must match the required observation/prediction time "
            f"grid; expected {expected_time_count} time points but got "
            f"{ctrl_time_count}."
        )

    # Avoid jnp.setxor1d/jnp.unique because their data-dependent output shapes
    # fail under JIT. Leading plate axes broadcast when one grid is shared.
    values_mismatch = ~jnp.allclose(
        jnp.sort(ctrl_times), jnp.sort(total_obs_pred_times)
    )
    _ = eqx.error_if(
        ctrl_times,
        values_mismatch,
        "Control times and the union of obs_times and predict_times must be the same.",
    )


def _build_control_path_eval(
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None,
    ctrl_values: (
        Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None
    ),
    obs_times: Real[Array, "*obs_time_plate obs_time"],
) -> Callable[[Real[Array, ""]], Real[Array, "..."] | None]:
    """
    Build a right-continuous control evaluator for continuous-time paths.

    Extends the path past the final time so that evaluate(t_last, left=False)
    returns the last value instead of NaN (rectilinear path has no right piece
    at the boundary).
    """
    if ctrl_times is None or ctrl_values is None:
        return lambda t: None

    t_final = jnp.maximum(obs_times[-1], ctrl_times[-1]) + _CONTROL_EXTEND_EPSILON
    ctrl_times_ext = jnp.concatenate([ctrl_times, t_final[None]])
    ctrl_values_ext = jnp.concatenate([ctrl_values, ctrl_values[-1:]], axis=0)
    _ct, _cv = dfx.rectilinear_interpolation(ts=ctrl_times_ext, ys=ctrl_values_ext)
    control_path = dfx.LinearInterpolation(ts=_ct, ys=_cv)
    return lambda t: control_path.evaluate(t, left=False)
