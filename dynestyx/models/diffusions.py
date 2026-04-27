"""Helpers for evaluating and normalizing continuous-time diffusion specs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple

import jax.numpy as jnp
from jax import Array

from dynestyx.types import Control, State, Time

DiffusionType = Literal["full", "diag", "scalar"]
DiffusionValue = Array | float | int
DiffusionSpec = Callable[[State, Control | None, Time], DiffusionValue] | DiffusionValue


class EvaluatedDiffusion(NamedTuple):
    diffusion_type: DiffusionType
    value: Array
    bm_dim: int


def evaluate_diffusion_value(
    diffusion_coefficient: DiffusionSpec,
    x: State,
    u: Control | None,
    t: Time,
) -> Array:
    """Evaluate a diffusion spec and return it as a JAX array."""
    value = (
        diffusion_coefficient(x, u, t)
        if callable(diffusion_coefficient)
        else diffusion_coefficient
    )
    return jnp.asarray(value)


def resolve_diffusion_metadata(
    shape: tuple[int, ...],
    *,
    state_dim: int,
    diffusion_type: DiffusionType | None,
    bm_dim: int | None,
) -> tuple[DiffusionType, int]:
    """Resolve diffusion semantics from trailing shape and user annotations."""
    resolved_type = (
        diffusion_type
        if diffusion_type is not None
        else _infer_diffusion_type(shape, state_dim=state_dim)
    )

    if resolved_type == "full":
        if len(shape) < 2 or int(shape[-2]) != state_dim:
            raise ValueError(
                "Full diffusion must have trailing shape (..., state_dim, bm_dim). "
                f"Got shape {shape} with state_dim={state_dim}."
            )
        inferred_bm_dim = int(shape[-1])
        if bm_dim is not None and int(bm_dim) != inferred_bm_dim:
            raise ValueError(
                "bm_dim does not match inferred diffusion_coefficient output shape. "
                f"Got bm_dim={bm_dim}, inferred={inferred_bm_dim} from shape {shape}."
            )
        return resolved_type, inferred_bm_dim

    if resolved_type == "diag":
        if len(shape) == 0 or int(shape[-1]) != state_dim:
            raise ValueError(
                "Diagonal diffusion must have trailing shape (..., state_dim). "
                f"Got shape {shape} with state_dim={state_dim}."
            )
    else:
        if len(shape) != 0 and int(shape[-1]) != 1:
            raise ValueError(
                "Scalar diffusion must have shape () or trailing shape (..., 1). "
                f"Got shape {shape}."
            )

    if bm_dim is None:
        raise ValueError(
            f"{resolved_type} diffusion requires explicit bm_dim. "
            "For scalar or diagonal diffusion, bm_dim must be either 1 or state_dim."
        )
    resolved_bm_dim = int(bm_dim)
    if resolved_bm_dim not in (1, state_dim):
        raise ValueError(
            f"{resolved_type} diffusion requires bm_dim to be either 1 or state_dim. "
            f"Got bm_dim={resolved_bm_dim}, state_dim={state_dim}."
        )
    return resolved_type, resolved_bm_dim


def evaluate_diffusion(
    diffusion_coefficient: DiffusionSpec,
    *,
    diffusion_type: DiffusionType | None,
    bm_dim: int | None,
    x: State,
    u: Control | None,
    t: Time,
    state_dim: int,
) -> EvaluatedDiffusion:
    """Evaluate a diffusion spec and resolve its semantics."""
    value = evaluate_diffusion_value(diffusion_coefficient, x, u, t)
    resolved_type, resolved_bm_dim = resolve_diffusion_metadata(
        value.shape,
        state_dim=state_dim,
        diffusion_type=diffusion_type,
        bm_dim=bm_dim,
    )
    return EvaluatedDiffusion(
        diffusion_type=resolved_type,
        value=value,
        bm_dim=resolved_bm_dim,
    )


def diffusion_as_matrix(
    diffusion: EvaluatedDiffusion,
    *,
    state_dim: int,
) -> Array:
    """Convert a structured diffusion into dense (..., state_dim, bm_dim) form."""
    value = diffusion.value
    if diffusion.diffusion_type == "full":
        return value

    eye = jnp.eye(state_dim, dtype=value.dtype)
    if diffusion.diffusion_type == "diag":
        if diffusion.bm_dim == 1:
            return value[..., :, None]
        return value[..., :, None] * eye

    scalar = value if value.ndim == 0 else jnp.squeeze(value, axis=-1)
    if diffusion.bm_dim == 1:
        return jnp.broadcast_to(scalar[..., None, None], scalar.shape + (state_dim, 1))
    return scalar[..., None, None] * eye


def diffusion_covariance(
    diffusion: EvaluatedDiffusion,
    *,
    state_dim: int,
) -> Array:
    """Return L L^T while preserving scalar/diagonal structure when possible."""
    value = diffusion.value
    if diffusion.diffusion_type == "full":
        return value @ jnp.swapaxes(value, -1, -2)

    eye = jnp.eye(state_dim, dtype=value.dtype)
    if diffusion.diffusion_type == "diag":
        if diffusion.bm_dim == 1:
            return value[..., :, None] * value[..., None, :]
        return jnp.square(value)[..., :, None] * eye

    sigma_sq = jnp.square(value if value.ndim == 0 else jnp.squeeze(value, axis=-1))
    if diffusion.bm_dim == 1:
        return sigma_sq[..., None, None] * jnp.ones(
            (state_dim, state_dim), dtype=value.dtype
        )
    return sigma_sq[..., None, None] * eye


def apply_diffusion(
    diffusion: EvaluatedDiffusion,
    dw: Array,
    *,
    state_dim: int,
) -> Array:
    """Apply a structured diffusion to a Brownian increment vector."""
    value = diffusion.value
    if diffusion.diffusion_type == "full":
        return value @ dw

    if diffusion.diffusion_type == "diag":
        if diffusion.bm_dim == 1:
            return value * dw[0]
        return value * dw

    scalar = value if value.ndim == 0 else jnp.squeeze(value, axis=-1)
    if diffusion.bm_dim == 1:
        return jnp.broadcast_to(scalar * dw[0], (state_dim,))
    return scalar * dw


def _infer_diffusion_type(
    shape: tuple[int, ...],
    *,
    state_dim: int,
) -> DiffusionType:
    """Infer legacy diffusion semantics from shape when diffusion_type is omitted."""
    if len(shape) >= 2 and int(shape[-2]) == state_dim:
        return "full"
    if len(shape) == 0 or int(shape[-1]) == 1:
        return "scalar"
    if int(shape[-1]) == state_dim:
        return "diag"
    trailing_dim = int(shape[-1])
    raise ValueError(
        f"1D diffusion output with trailing dimension {trailing_dim} is treated as "
        f"scalar/diagonal shorthand, so it must end in 1 or state_dim={state_dim}. "
        "Use shape (..., state_dim, bm_dim) for full diffusion."
    )
