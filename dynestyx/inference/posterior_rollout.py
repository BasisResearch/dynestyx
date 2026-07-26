"""Shared helpers for posterior-rollout time handling."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Real


def _validate_future_only_predict_times(
    predict_times: Real[Array, "*predict_time_plate predict_time"] | None,
    anchor_times: Real[Array, "*anchor_time_plate anchor_time"] | None,
    *,
    error_message: str,
) -> Real[Array, "*predict_time_plate predict_time"] | None:
    """Validate the future-only posterior-rollout contract."""
    if predict_times is None or anchor_times is None:
        return predict_times
    anchor_end = anchor_times[..., -1:]
    _ = eqx.error_if(
        predict_times,
        jnp.any(predict_times < anchor_end),
        error_message,
    )
    return predict_times


def _final_times_for_rollout(
    times: Real[Array, "*time_plate time"],
) -> Real[Array, "*time_plate one"]:
    """Return the final anchor time while keeping simulator segmentation host-safe."""
    try:
        times_host = np.asarray(jax.device_get(times))
        return jnp.asarray(times_host[..., -1:], dtype=times.dtype)
    except Exception:
        return times[..., -1:]
