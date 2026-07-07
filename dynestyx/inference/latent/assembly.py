"""Internal latent-state assembly helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import jax.numpy as jnp
from jax import Array

from dynestyx.inference.latent.metadata import (
    DiracLatentMetadata,
    canonicalize_dirac_state_path_params,
    canonicalize_state_path_params,
)
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.solvers import solve_ode
from dynestyx.utils import _build_control_path, _raise_now_or_error_if


@dataclasses.dataclass
class AssembledStateTrajectory:
    """Latent parameterization plus reconstructed full state trajectory.

    Mathematically, the probabilistic model is written in terms of a full state
    path

    ``x = (x_0, x_1, ..., x_T)``.

    This object separates that path from its free parameterization:

    - ``state_path_params`` are free variables ``z``,
    - ``state_path`` is the reconstructed state path ``x = g(z)``.

    The model density is then evaluated as ``log p(x, y)`` after reconstructing
    ``x`` from ``z``.

    These can be identical, but they do not have to be:
    - discrete / discretized v1: often identical,
    - deterministic continuous-time models: ``state_path_params`` are the
      initial condition, while ``state_path`` is the solved path,
    - compressed exact-observation layouts: ``state_path_params`` contain only
      the free coordinates, while ``state_path`` is the fully reconstructed
      state grid.
    """

    state_path_params: Array
    state_path_param_times: Array
    state_path_param_coordinate_indices: Array | None
    state_path: Array
    state_path_times: Array


def _ode_state_times(
    state_path_param_times: Array,
    obs_times: Array,
) -> Array:
    """Return the evaluation grid for an ODE latent-state assembly.

    We always prepend the initial-condition time so the reconstructed path has
    an explicit first state corresponding to the IC. If ``obs_times`` already
    starts at ``t0``, the returned grid contains that overlap twice; downstream
    observation scoring uses exact-time lookup and therefore still aligns the
    observations to the first occurrence at the IC.
    """
    obs_times_arr = jnp.asarray(obs_times)
    return jnp.concatenate([state_path_param_times, obs_times_arr], axis=0)


def default_ode_diffeqsolve_settings() -> dict[str, Any]:
    """Return the default ODE solve settings for latent-state assembly."""
    return {
        "solver": dfx.Tsit5(),
        "stepsize_controller": dfx.ConstantStepSize(),
        "adjoint": dfx.RecursiveCheckpointAdjoint(),
        "dt0": jnp.asarray(1e-3),
        "max_steps": 100_000,
    }


def assemble_dirac_state_path(
    *,
    state_path_params: Array,
    latent_metadata: DiracLatentMetadata,
    obs_times: Array,
    obs_values_filled: Array,
) -> AssembledStateTrajectory:
    """Reconstruct full state values from compressed exact-observation latents.

    If ``z = state_path_params`` denotes only the free coordinates, then the
    returned ``state_path`` is the full reconstructed path ``x = g(z)`` on the
    observation grid after reinserting observed coordinates.
    """
    canonical_params = canonicalize_dirac_state_path_params(
        state_path_params,
        n_state_path_params=latent_metadata.free_flat_indices.shape[0],
    )
    flat_state_path = jnp.reshape(jnp.asarray(obs_values_filled), (-1,))
    state_path = (
        flat_state_path.at[latent_metadata.free_flat_indices]
        .set(canonical_params)
        .reshape(latent_metadata.state_shape)
    )
    obs_times_arr = jnp.asarray(obs_times)
    return AssembledStateTrajectory(
        state_path_params=canonical_params,
        state_path_param_times=latent_metadata.state_path_param_times,
        state_path_param_coordinate_indices=(
            latent_metadata.state_path_param_coordinate_indices
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
) -> AssembledStateTrajectory:
    """Assemble full state values from a latent-state parameterization.

    This is the standard assembly path for non-compressed latent
    parameterizations:
    - discrete / discretized models: the path params are already a full path,
    - deterministic continuous-time models: the path params specify only the
      initial condition and the ODE solve reconstructs the path.
    """
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
            state_path_times = _ode_state_times(state_path_param_times, obs_times)
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

        return AssembledStateTrajectory(
            state_path_params=canonical_params,
            state_path_param_times=state_path_param_times,
            state_path_param_coordinate_indices=None,
            state_path=state_path,
            state_path_times=state_path_times,
        )

    return AssembledStateTrajectory(
        state_path_params=canonical_params,
        state_path_param_times=state_path_param_times,
        state_path_param_coordinate_indices=None,
        state_path=canonical_params,
        state_path_times=state_path_param_times,
    )


__all__ = [
    "AssembledStateTrajectory",
    "assemble_dirac_state_path",
    "assemble_state_path",
    "default_ode_diffeqsolve_settings",
]
