"""Validation and shape-inference helpers for dynamical models."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from dynestyx.types import Control, State, Time


def _is_categorical_distribution(distribution: Any) -> bool:
    return hasattr(
        distribution, "probs"
    ) and distribution.__class__.__name__.startswith("Categorical")


def _infer_vector_dim_from_distribution(distribution: Any, name: str) -> int:
    if _is_categorical_distribution(distribution):
        return int(jnp.asarray(distribution.probs).shape[-1])

    shape: tuple[int, ...] | None = None

    if hasattr(distribution, "shape"):
        try:
            shape = tuple(int(d) for d in distribution.shape())
        except TypeError:
            shape = tuple(int(d) for d in distribution.shape(()))

    if shape is None:
        if hasattr(distribution, "batch_shape") and hasattr(
            distribution, "event_shape"
        ):
            shape = tuple(distribution.batch_shape) + tuple(distribution.event_shape)
        else:
            raise ValueError(
                f"Could not infer shape from {name}: object has no shape metadata."
            )

    if len(shape) == 0:
        return 1
    if len(shape) == 1:
        return int(shape[0])
    raise ValueError(
        f"{name} must have scalar or vector support shape; got shape {shape}."
    )


def _make_probe_state(initial_condition: Any, state_dim: int) -> jax.Array:
    if _is_categorical_distribution(initial_condition):
        return jnp.array(0, dtype=jnp.int32)
    return jnp.zeros((state_dim,))


def _validate_continuous_state_evolution(
    state_evolution: Any,
    state_dim: int,
    x0: State,
    u0: Control | None,
    t0: Time,
) -> None:
    drift_shape = jax.eval_shape(lambda: state_evolution.total_drift(x0, u0, t0)).shape
    if drift_shape != (state_dim,):
        raise ValueError(
            "State drift shape is inconsistent with state_dim. "
            f"Expected {(state_dim,)}, got {drift_shape}."
        )

    if state_evolution.diffusion_coefficient is not None:
        diffusion_shape = jax.eval_shape(
            lambda: state_evolution.diffusion_coefficient(x0, u0, t0)
        ).shape
        if len(diffusion_shape) != 2:
            raise ValueError(
                "diffusion_coefficient must return a matrix with shape "
                "(state_dim, bm_dim). "
                f"Got shape {diffusion_shape}."
            )
        if diffusion_shape[0] != state_dim:
            raise ValueError(
                "diffusion_coefficient first dimension must match state_dim. "
                f"Got diffusion shape {diffusion_shape}, state_dim={state_dim}."
            )
        state_evolution.bm_dim = int(diffusion_shape[1])
    else:
        state_evolution.bm_dim = None


def _validate_state_evolution_output_shape(
    state_evolution: Callable[[State, Control, Time], State]
    | Callable[[State, Control, Time, Time], State],
    state_dim: int,
    x0: State,
    u0: Control | None,
    t0: Time,
    *,
    continuous_time: bool,
) -> None:
    if continuous_time:
        _validate_continuous_state_evolution(
            state_evolution=state_evolution,
            state_dim=state_dim,
            x0=x0,
            u0=u0,
            t0=t0,
        )
    else:
        t_now = t0
        t_next = t0 + 1.0
        transition_dist = state_evolution(x=x0, u=u0, t_now=t_now, t_next=t_next)  # type: ignore[misc,call-arg]
        inferred_state_dim = _infer_vector_dim_from_distribution(
            transition_dist, "state_evolution(x, u, t_now, t_next)"
        )
        if inferred_state_dim != state_dim:
            raise ValueError(
                "State transition shape is inconsistent with state_dim. "
                f"state_dim={state_dim}, inferred={inferred_state_dim}."
            )
