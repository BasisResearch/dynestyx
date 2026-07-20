"""Pure-JAX state-path reconstruction helpers.

This module defines how latent path parameters ``z = state_path_params`` are
turned into a full state trajectory ``x = state_path = g(z)``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    assemble_completed_observations,
    validate_missing_obs_values,
)
from dynestyx.solvers import solve_ode_state_path
from dynestyx.utils import _raise_now_or_error_if


def validate_state_path_params(
    dynamics: DynamicalModel,
    state_path_params: Array,
    *,
    n_times: int,
) -> Array:
    """Validate state-path parameters and return them with a leading time axis."""
    params = jnp.asarray(state_path_params)
    event_ndim = len(dynamics.initial_condition.event_shape)

    if n_times == 1:
        if params.ndim == event_ndim:
            return jnp.expand_dims(params, axis=0)
        if params.ndim == event_ndim + 1 and params.shape[0] == 1:
            return params
        raise ValueError(
            "state_path_params is incompatible with state_path_param_times. "
            "For a single parameter time, provide either one path parameter or a "
            "length-1 leading time axis."
        )

    if params.ndim < 1 or params.shape[0] != n_times:
        raise ValueError(
            "state_path_params must have a leading time axis matching "
            "state_path_param_times for discrete / discretized models."
        )
    return params


def infer_state_path_param_times(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
) -> Array:
    """Infer the time index attached to ``state_path_params``."""
    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if dynamics.t0 is None:
            raise ValueError(
                "Deterministic continuous-time latent-state assembly requires "
                "dynamics.t0 to be resolved before inferring path parameter times."
            )
        obs_times_arr = jnp.asarray(obs_times)
        return jnp.asarray([jnp.asarray(dynamics.t0, dtype=obs_times_arr.dtype)])
    return jnp.asarray(obs_times)


def reconstruct_state_path_from_exact_observations(
    *,
    state_path_params: Array,
    latent_metadata: MissingObservationMetadata,
    obs_times: Array,
    obs_values_filled: Array,
) -> tuple[Array, Array, Array]:
    """Fill missing exact observations to reconstruct the state trajectory."""
    try:
        validated_params = validate_missing_obs_values(
            state_path_params,
            n_missing_obs=latent_metadata.missing_flat_indices.shape[0],
        )
    except ValueError as exc:
        raise ValueError(
            "Exact-observation state_path_params must be a flat vector whose "
            "length matches the number of missing state coordinates."
        ) from exc
    state_path = assemble_completed_observations(
        obs_values_filled=jnp.asarray(obs_values_filled),
        missing_obs_values=validated_params,
        missing_obs_metadata=latent_metadata,
    )
    obs_times_arr = jnp.asarray(obs_times)
    return validated_params, state_path, obs_times_arr


def reconstruct_state_path(
    dynamics: DynamicalModel,
    *,
    state_path_params: Array,
    state_path_param_times: Array,
    obs_times: Array | None = None,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    ode_diffeqsolve_settings: dict[str, Any] | None = None,
) -> tuple[Array, Array, Array]:
    """Reconstruct a discrete path or solve an ODE from its initial state."""
    state_path_param_times = jnp.asarray(state_path_param_times)
    _raise_now_or_error_if(
        state_path_param_times,
        state_path_param_times.shape[0] < 1,
        "state_path_param_times must contain at least one time point.",
    )

    validated_params = validate_state_path_params(
        dynamics,
        state_path_params,
        n_times=state_path_param_times.shape[0],
    )

    if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
        raise ValueError(
            "Latent-state assembly does not yet support native SDE models. "
            "Please discretize the model first."
        )

    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if state_path_param_times.shape[0] != 1:
            raise ValueError(
                "Deterministic continuous-time models expect exactly one latent "
                "path parameter: the initial condition."
            )

        if obs_times is None:
            state_path_times = state_path_param_times
            state_path = validated_params
        else:
            state_path_times = jnp.concatenate(
                [state_path_param_times, jnp.asarray(obs_times)],
                axis=0,
            )
            state_path = solve_ode_state_path(
                dynamics,
                t0=state_path_param_times[0],
                initial_state=validated_params[0],
                path_times=state_path_times,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                diffeqsolve_settings=ode_diffeqsolve_settings,
            )

        return validated_params, state_path, state_path_times

    return validated_params, validated_params, state_path_param_times


__all__ = [
    "reconstruct_state_path",
    "reconstruct_state_path_from_exact_observations",
    "infer_state_path_param_times",
    "validate_state_path_params",
]
