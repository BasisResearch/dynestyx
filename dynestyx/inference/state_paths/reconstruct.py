"""Pure-JAX state-path reconstruction helpers.

This module defines how latent path parameters ``z = state_path_params`` are
turned into a full state trajectory ``x = state_path = g(z)``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax as dfx
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
    canonicalize_missing_obs_values,
)
from dynestyx.solvers import solve_ode
from dynestyx.utils import _build_control_path, _raise_now_or_error_if


@dataclasses.dataclass
class AssembledStatePath:
    """Reconstructed state path ``x = g(z)`` for one parameterization."""

    state_path_params: Array
    state_path_param_times: Array
    state_path_param_coordinate_indices: Array | None
    state_path: Array
    state_path_times: Array


def canonicalize_state_path_params(
    dynamics: DynamicalModel,
    state_path_params: Array,
    *,
    n_times: int,
) -> Array:
    """Canonicalize dense ``state_path_params`` so time is the leading axis."""
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


def canonicalize_completed_observation_state_params(
    state_path_params: Array,
    *,
    n_state_path_params: int,
) -> Array:
    """Canonicalize exact-observation path params to a flat free-coordinate vector."""
    try:
        return canonicalize_missing_obs_values(
            state_path_params,
            n_missing_obs=n_state_path_params,
        )
    except ValueError as exc:
        raise ValueError(
            "Completed-observation state_path_params must be a flat vector "
            "whose length matches the number of free state coordinates."
        ) from exc


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


def default_ode_diffeqsolve_settings() -> dict[str, Any]:
    """Return default solver settings for deterministic ODE path reconstruction."""
    return {
        "solver": dfx.Tsit5(),
        "stepsize_controller": dfx.ConstantStepSize(),
        "adjoint": dfx.RecursiveCheckpointAdjoint(),
        "dt0": jnp.asarray(1e-3),
        "max_steps": 100_000,
    }


def assemble_completed_observation_state_path(
    *,
    state_path_params: Array,
    latent_metadata: MissingObservationMetadata,
    obs_times: Array,
    obs_values_filled: Array,
) -> AssembledStatePath:
    """Reconstruct the state path from completed exact observations."""
    canonical_params = canonicalize_completed_observation_state_params(
        state_path_params,
        n_state_path_params=latent_metadata.free_flat_indices.shape[0],
    )
    state_path = assemble_completed_observations(
        obs_values_filled=jnp.asarray(obs_values_filled),
        missing_obs_values=canonical_params,
        missing_obs_metadata=latent_metadata,
    )
    obs_times_arr = jnp.asarray(obs_times)
    return AssembledStatePath(
        state_path_params=canonical_params,
        state_path_param_times=latent_metadata.missing_obs_times,
        state_path_param_coordinate_indices=(
            latent_metadata.missing_obs_coordinate_indices
        ),
        state_path=state_path,
        state_path_times=obs_times_arr,
    )


def assemble_state_path(
    dynamics: DynamicalModel,
    *,
    state_path_params: Array,
    state_path_param_times: Array,
    obs_times: Array | None = None,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    ode_diffeqsolve_settings: dict[str, Any] | None = None,
) -> AssembledStatePath:
    """Assemble a full state path ``x = g(z)`` from dense path parameters."""
    state_path_param_times = jnp.asarray(state_path_param_times)
    _raise_now_or_error_if(
        state_path_param_times,
        state_path_param_times.shape[0] < 1,
        "state_path_param_times must contain at least one time point.",
    )

    canonical_params = canonicalize_state_path_params(
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
            state_path = canonical_params
        else:
            state_path_times = jnp.concatenate(
                [state_path_param_times, jnp.asarray(obs_times)],
                axis=0,
            )
            if ctrl_times is not None and ctrl_values is not None:
                control_path = _build_control_path(
                    ctrl_times, ctrl_values, state_path_times
                )
                control_path_eval: Callable[[Array], Array | None] = lambda t: (
                    control_path.evaluate(t, left=False)
                )
            else:
                control_path_eval = lambda t: None

            state_path = solve_ode(
                dynamics,
                t0=state_path_param_times[0],
                saveat_times=state_path_times,
                x0=canonical_params[0],
                control_path_eval=control_path_eval,
                diffeqsolve_settings=(
                    ode_diffeqsolve_settings
                    if ode_diffeqsolve_settings is not None
                    else default_ode_diffeqsolve_settings()
                ),
            )

        return AssembledStatePath(
            state_path_params=canonical_params,
            state_path_param_times=state_path_param_times,
            state_path_param_coordinate_indices=None,
            state_path=state_path,
            state_path_times=state_path_times,
        )

    return AssembledStatePath(
        state_path_params=canonical_params,
        state_path_param_times=state_path_param_times,
        state_path_param_coordinate_indices=None,
        state_path=canonical_params,
        state_path_times=state_path_param_times,
    )


__all__ = [
    "AssembledStatePath",
    "assemble_completed_observation_state_path",
    "assemble_state_path",
    "canonicalize_completed_observation_state_params",
    "canonicalize_state_path_params",
    "default_ode_diffeqsolve_settings",
    "infer_state_path_param_times",
]
