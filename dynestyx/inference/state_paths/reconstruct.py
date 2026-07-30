"""Pure-JAX state-path reconstruction helpers.

This module defines how latent path parameters ``z = state_path_params`` are
turned into a full state trajectory ``x = state_path = g(z)``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Real

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
    state_path_params: Real[Array, "state_path_param_time state_dim"]
    | Real[Array, " _"]
    | Real[Array, ""],
    *,
    n_times: int,
) -> (
    Real[Array, "state_path_param_time state_dim"]
    | Real[Array, " state_path_param_time"]
):
    """Validate the time axis of state-path parameters.

    If `n_times == 1`, this function accepts either one state value (determined
    by `dynamics.initial_condition`) or an array with a length-one leading
    time axis. The returned states are normalized to have a time-leading axis.

    For `n_times > 1`, `state_path_params` must have a leading axis
    of length `n_times`. This function does not compare the remaining dimensions
    with the sizes in `initial_condition.event_shape`.

    Args:
        dynamics: Dynamical model that defines the number of state event axes.
        state_path_params: State values to validate.
        n_times: Expected number of entries on the leading time axis.

    Returns:
        Array: State-path parameters with a leading time axis.

    Raises:
        ValueError: If a single-time value has the wrong number of axes, or if
            an existing leading time axis does not have length `n_times`.
    """
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
    obs_times: Real[Array, " obs_time"],
) -> Real[Array, " state_path_param_time"]:
    """Return the times associated with `state_path_params`.

    A deterministic ODE has one state-path parameter: its initial state at
    `dynamics.t0`. All other models use one state-path parameter at each
    observation time.

    Args:
        dynamics: Dynamical model that determines the path representation.
        obs_times: Observation times. The returned ODE time uses the same
            dtype.

    Returns:
        Array: A length-one array containing `dynamics.t0` for a deterministic
            ODE, or `obs_times` converted to a JAX array for any other model.

    Raises:
        ValueError: If a deterministic ODE does not define `dynamics.t0`.
    """
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
    state_path_params: Real[Array, " n_missing_state"] | Real[Array, ""],
    latent_metadata: MissingObservationMetadata,
    obs_times: Real[Array, " obs_time"],
    obs_values_filled: Real[Array, "obs_time state_dim"] | Real[Array, " obs_time"],
) -> tuple[
    Real[Array, " n_missing_state"],
    Real[Array, "obs_time state_dim"] | Real[Array, " obs_time"],
    Real[Array, " obs_time"],
]:
    """Reconstruct a state path from exact identity observations.

    With `DiracIdentityObservation`, each observed value is also a state value.
    `state_path_params` supplies only the missing state components. The
    positions and order of those components are given by
    `latent_metadata.missing_flat_indices`.

    Validation requires one parameter for each missing component. No missing
    components require an empty array. One missing component accepts either a
    scalar or a length-one vector. Two or more missing components require a flat
    vector with the matching length.

    Args:
        state_path_params: Values used to fill the missing state components.
        latent_metadata: Missing-component positions and the completed path
            shape.
        obs_times: Observation times, which are also the state-path times.
        obs_values_filled: Observation array with placeholder values at missing
            positions.

    Returns:
        tuple[Array, Array, Array]: A tuple containing the validated flat
            parameter vector, the completed state path, and the state-path
            times.

    Raises:
        ValueError: If the number or shape of `state_path_params` does not match
            the missing components recorded in `latent_metadata`.
    """
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
    state_path_params: Real[Array, "state_path_param_time state_dim"]
    | Real[Array, " _"]
    | Real[Array, ""],
    state_path_param_times: Real[Array, " state_path_param_time"],
    obs_times: Real[Array, " obs_time"] | None = None,
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None = None,
    ode_diffeqsolve_settings: dict[str, Any] | None = None,
) -> tuple[
    Real[Array, "state_path_param_time state_dim"]
    | Real[Array, " state_path_param_time"],
    Real[Array, "state_path_time state_dim"] | Real[Array, " state_path_time"],
    Real[Array, " state_path_time"],
]:
    """Reconstruct a complete state path from its parameter values.

    For a discrete or discretized model, `state_path_params` is already the
    complete path. Its leading axis must match `state_path_param_times`.

    For a deterministic ODE, `state_path_params` contains one initial state and
    `state_path_param_times` contains its time. If `obs_times` is provided, the
    function solves the ODE on the concatenated time array
    `[state_path_param_times, obs_times]`. If `obs_times` is `None`, the
    returned path contains only the initial state.

    Args:
        dynamics: Dynamical model that defines the state evolution.
        state_path_params: State values used to construct the path.
        state_path_param_times: Times associated with `state_path_params`. This
            array must contain at least one entry.
        obs_times: Observation times used as ODE solution times. Discrete models
            do not use this argument.
        ctrl_times: Times associated with `ctrl_values`.
        ctrl_values: Control values passed to the ODE solver, or `None` for an
            uncontrolled model.
        ode_diffeqsolve_settings: Settings passed to the Diffrax ODE solver.

    Returns:
        tuple[Array, Array, Array]: A tuple containing the validated parameters,
            the complete state path, and the times associated with the complete
            path.

    Raises:
        ValueError: If `state_path_param_times` is empty, the parameter time
            axis has an incompatible length, a deterministic ODE has more than
            one parameter time, or `dynamics` is a continuous-time SDE that has
            not been discretized.
    """
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
