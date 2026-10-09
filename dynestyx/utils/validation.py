"""Shared numeric and time-grid validation with JAX-compatible error handling."""

import math
import warnings
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Bool, Real, Shaped


def _raise_now_or_error_if(
    anchor: Shaped[Array, "*shape"],
    predicate: Bool[Array, ""] | bool,
    message: str,
    *,
    action: Literal["raise", "warn"] = "raise",
) -> Shaped[Array, "*shape"]:
    """Raise or warn for a predicate, returning the anchor to preserve JIT checks.

    Warnings are emitted only for eager predicates. A traced warning predicate
    returns the anchor unchanged rather than introducing a JAX runtime callback.
    """
    try:
        should_handle = bool(predicate)
    except jax.errors.TracerBoolConversionError:
        if action == "raise":
            return eqx.error_if(anchor, predicate, message)
        return anchor

    if not should_handle:
        return anchor

    if action == "warn":
        warnings.warn(message, stacklevel=2)
        return anchor

    if action == "raise":
        raise ValueError(message)

    raise AssertionError(f"Unexpected action for _raise_now_or_error_if: {action!r}")


def _validate_nonnegative_float(name: str, value: float) -> None:
    """Validate a nonnegative, finite float-valued config field.

    Shared by the jitter fields on the discretizer and filter configs, which
    all have the same admissible range. ``name`` is the field's own name, so
    the error message points at the attribute the user actually set.

    Args:
        name: Name of the field being validated, as it appears on the config.
        value: Value to validate.

    Raises:
        ValueError: If ``value`` is not finite or is negative.
    """
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, nonnegative float, got {value!r}.")


def _validate_site_sorting(
    times: Real[Array, "*time_plate time"] | None, name: str
) -> None:
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
