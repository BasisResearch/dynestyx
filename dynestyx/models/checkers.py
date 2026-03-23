"""Validation and shape-inference helpers for dynamical models."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import numpyro.primitives

from dynestyx.types import Control, State, Time


def _unwrap_base_distribution(distribution: Any) -> Any:
    """Peel common NumPyro wrapper distributions to inspect the base distribution.

    NumPyro often wraps scalar/vector distributions in containers like
    `Independent`, `ExpandedDistribution`, or `MaskedDistribution`. For shape and
    categorical checks we want to reason about the base distribution semantics.
    """
    current = distribution
    while hasattr(current, "base_dist"):
        current = current.base_dist
    return current


def _is_categorical_distribution(distribution: Any) -> bool:
    """Return True for class-label categorical distributions.

    This intentionally excludes one-hot categorical variants because the model
    logic here assumes scalar integer latent states.
    """
    base = _unwrap_base_distribution(distribution)
    if isinstance(base, (dist.CategoricalProbs, dist.CategoricalLogits)):
        return True
    # Fallback for compatibility with custom/aliased categorical classes.
    name = base.__class__.__name__
    return name.startswith("Categorical") and "OneHot" not in name


def _infer_vector_dim_from_distribution(distribution: Any, name: str) -> int:
    """Infer scalar/vector dimension from a NumPyro-compatible distribution."""
    if _is_categorical_distribution(distribution):
        base = _unwrap_base_distribution(distribution)
        return int(jnp.asarray(base.probs).shape[-1])

    shape: tuple[int, ...] | None = None

    if hasattr(distribution, "shape"):
        try:
            # NumPyro distribution.shape() usually accepts no args, while some
            # wrappers/proxies expect a sample-shape positional argument.
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
    """Build a synthetic state value used for shape-check probes."""
    if _is_categorical_distribution(initial_condition):
        return jnp.array(0, dtype=jnp.int32)
    return jnp.zeros((state_dim,))


def _validate_continuous_state_evolution(
    state_evolution: Any,
    state_dim: int,
    x0: State,
    u0: Control | None,
    t0: Time,
) -> int | None:
    """Validate the shape of the continuous-time state evolution w.r.t. state_dim and bm_dim.

    Returns the inferred bm_dim (or None if no diffusion coefficient).
    """
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
        inferred_bm_dim = int(diffusion_shape[1])
        if (
            state_evolution.bm_dim is not None
            and int(state_evolution.bm_dim) != inferred_bm_dim
        ):
            raise ValueError(
                "bm_dim does not match inferred diffusion_coefficient output shape. "
                f"Got bm_dim={state_evolution.bm_dim}, inferred={inferred_bm_dim}."
            )
        return inferred_bm_dim
    else:
        if state_evolution.bm_dim is not None:
            raise ValueError("bm_dim cannot be set when diffusion_coefficient is None.")
        return None


def _validate_state_evolution_output_shape(
    state_evolution: Callable[[State, Control, Time], State]
    | Callable[[State, Control, Time, Time], State],
    state_dim: int,
    x0: State,
    u0: Control | None,
    t0: Time,
    *,
    continuous_time: bool,
) -> int | None:
    """Validate the shape of the state evolution w.r.t. state_dim (and bm_dim for continuous-time models).

    Returns the inferred bm_dim for continuous-time models, or None otherwise.
    """
    if continuous_time:
        return _validate_continuous_state_evolution(
            state_evolution=state_evolution,
            state_dim=state_dim,
            x0=x0,
            u0=u0,
            t0=t0,
        )
    else:
        if getattr(state_evolution, "bm_dim", None) is not None:
            raise ValueError(
                "bm_dim can only be set for continuous-time models with "
                "diffusion_coefficient."
            )
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
        return None


def _validate_continuous_time_flag(
    continuous_time: bool | None, inferred_continuous_time: bool
) -> None:
    """Ensure optional continuous_time agrees with inferred model type."""
    if continuous_time is not None and bool(continuous_time) != inferred_continuous_time:
        raise ValueError("continuous_time does not match inferred state_evolution type.")


def _validate_state_dim(state_dim: int | None, inferred_state_dim: int) -> None:
    """Ensure optional state_dim agrees with inferred initial condition shape."""
    if state_dim is not None and int(state_dim) != int(inferred_state_dim):
        raise ValueError(
            "state_dim does not match inferred initial_condition shape. "
            f"Got state_dim={state_dim}, inferred={inferred_state_dim}."
        )


def _validate_categorical_state(
    categorical_state: bool | None, inferred_categorical_state: bool
) -> None:
    """Ensure optional categorical_state agrees with inferred initial condition type."""
    if (
        categorical_state is not None
        and bool(categorical_state) != inferred_categorical_state
    ):
        raise ValueError(
            "categorical_state does not match inferred initial_condition type. "
            f"Got categorical_state={categorical_state}, "
            f"inferred={inferred_categorical_state}."
        )


def _inside_numpyro_plate_context() -> bool:
    """Return True when currently executing inside any active numpyro.plate frame."""
    return any(
        isinstance(frame, numpyro.primitives.plate)
        for frame in numpyro.primitives._PYRO_STACK
    )


def _infer_observation_dim_in_plate_context(
    *,
    initial_condition: Any,
    observation_model: Callable[[State, Control | None, Time], Any],
    inferred_state_dim: int,
    control_dim: int,
    t0: float | None,
    observation_dim: int | None,
) -> int:
    """Infer observation dimension in plate context, falling back to explicit value."""
    if observation_dim is not None:
        return int(observation_dim)

    x0 = _make_probe_state(
        initial_condition=initial_condition,
        state_dim=inferred_state_dim,
    )
    u0 = None if control_dim == 0 else jnp.zeros((control_dim,))
    dummy_t0 = jnp.array(0.0) if t0 is None else jnp.array(t0)
    try:
        obs_dist = observation_model(x0, u0, dummy_t0)
        return int(
            _infer_vector_dim_from_distribution(obs_dist, "observation_model(x, u, t)")
        )
    except Exception:
        return 0


def _validate_observation_dim(
    observation_dim: int | None, inferred_observation_dim: int
) -> None:
    """Ensure optional observation_dim agrees with inferred observation shape."""
    if observation_dim is not None and int(observation_dim) != int(
        inferred_observation_dim
    ):
        raise ValueError(
            "observation_dim does not match inferred observation_model output shape. "
            f"Got observation_dim={observation_dim}, inferred={inferred_observation_dim}."
        )
