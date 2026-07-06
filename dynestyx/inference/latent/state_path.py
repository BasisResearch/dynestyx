"""Internal latent-state assembly helpers."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.observation_missingness import prepare_observation_views
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


@dataclasses.dataclass
class DiracLatentMetadata:
    """Concrete indexing metadata for exact-observation latent compression."""

    state_path_param_times: Array
    state_path_param_coordinate_indices: Array
    free_flat_indices: Array
    state_shape: tuple[int, ...]


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


def canonicalize_state_path_params(
    dynamics: DynamicalModel,
    state_path_params: Array,
    *,
    n_times: int,
) -> Array:
    """Canonicalize path params so the leading axis indexes parameter times."""
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


def canonicalize_dirac_state_path_params(
    state_path_params: Array,
    *,
    n_state_path_params: int,
) -> Array:
    """Canonicalize compressed exact-observation path params to a flat vector."""
    params = jnp.asarray(state_path_params)

    if n_state_path_params == 0:
        if params.size != 0:
            raise ValueError(
                "This exact-observation trajectory has no free state_path_params. "
                "Provide an empty state_path_params vector."
            )
        return jnp.reshape(params, (0,))

    if params.ndim == 0 and n_state_path_params == 1:
        return jnp.reshape(params, (1,))

    if params.ndim != 1 or params.shape[0] != n_state_path_params:
        raise ValueError(
            "Compressed exact-observation state_path_params must be a flat vector "
            "whose length matches the number of free state coordinates."
        )
    return params


def infer_state_path_param_times(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
) -> Array:
    """Infer parameter times for the current v1 path parameterization."""
    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if dynamics.t0 is None:
            raise ValueError(
                "Deterministic continuous-time latent-state assembly requires "
                "dynamics.t0 to be resolved before inferring path parameter times."
            )
        obs_times_arr = jnp.asarray(obs_times)
        return jnp.asarray([jnp.asarray(dynamics.t0, dtype=obs_times_arr.dtype)])
    return jnp.asarray(obs_times)


def infer_dirac_state_path_param_metadata(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_mask: Array,
) -> DiracLatentMetadata:
    """Infer compressed latent indexing for discrete exact observations."""
    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        raise ValueError(
            "DiracIdentityObservation missingness compression is not yet "
            "implemented for deterministic continuous-time models."
        )
    if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
        raise ValueError(
            "Latent-state assembly does not yet support native SDE models. "
            "Please discretize the model first."
        )
    if not isinstance(dynamics.observation_model, DiracIdentityObservation):
        raise ValueError(
            "Dirac latent metadata is only defined for DiracIdentityObservation."
        )

    try:
        obs_mask_np = np.asarray(obs_mask, dtype=bool)
        obs_times_np = np.asarray(obs_times)
    except Exception as exc:  # pragma: no cover - defensive for traced callers
        raise ValueError(
            "Dirac latent compression currently requires a concrete observation "
            "missingness pattern. Precompute it eagerly with "
            "prepare_dirac_state_path_metadata(...) and pass the result to "
            "LatentPathBuilder(dirac_state_path_metadata=...)."
        ) from exc

    free_mask_np = ~obs_mask_np
    flat_free_indices_np = np.flatnonzero(free_mask_np.reshape(-1))

    if obs_mask_np.ndim == 1:
        state_path_param_times_np = obs_times_np[free_mask_np]
        coord_indices_np = np.zeros((flat_free_indices_np.shape[0],), dtype=np.int32)
    elif obs_mask_np.ndim == 2:
        time_grid_np = np.broadcast_to(obs_times_np[:, None], obs_mask_np.shape)
        coord_grid_np = np.broadcast_to(
            np.arange(obs_mask_np.shape[-1], dtype=np.int32)[None, :],
            obs_mask_np.shape,
        )
        state_path_param_times_np = time_grid_np[free_mask_np]
        coord_indices_np = coord_grid_np[free_mask_np]
    else:
        raise ValueError(
            "Dirac latent compression expects obs_mask with shape (time,) or "
            "(time, observation_dim)."
        )

    obs_times_arr = jnp.asarray(obs_times)
    return DiracLatentMetadata(
        state_path_param_times=jnp.asarray(
            state_path_param_times_np,
            dtype=obs_times_arr.dtype,
        ),
        state_path_param_coordinate_indices=jnp.asarray(
            coord_indices_np, dtype=jnp.int32
        ),
        free_flat_indices=jnp.asarray(flat_free_indices_np, dtype=jnp.int32),
        state_shape=tuple(obs_mask_np.shape),
    )


def prepare_dirac_state_path_metadata(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_values: Array | None = None,
    obs_mask: Array | None = None,
) -> DiracLatentMetadata:
    """Precompute exact-observation compression metadata outside JIT/MCMC traces.

    This is the stable path for partial-missing `DiracIdentityObservation`
    models under NumPyro inference. The missingness pattern determines how many
    free ``state_path_params`` are needed, so callers should prepare this
    metadata eagerly from concrete observations before entering traced MCMC/SVI
    code.
    """
    if (obs_values is None) == (obs_mask is None):
        raise ValueError(
            "Provide exactly one of obs_values or obs_mask when preparing Dirac "
            "state-path metadata."
        )

    if obs_mask is None:
        assert obs_values is not None
        _, obs_mask, _ = prepare_observation_views(dynamics, obs_values)
        if obs_mask is None:
            raise ValueError(
                "Could not prepare an observation mask for Dirac state-path metadata."
            )

    return infer_dirac_state_path_param_metadata(
        dynamics,
        obs_times=obs_times,
        obs_mask=obs_mask,
    )


def fully_observed_dirac_state_path_param_metadata(
    *,
    obs_times: Array,
    state_shape: tuple[int, ...],
) -> DiracLatentMetadata:
    """Return empty compression metadata for fully observed exact observations."""
    obs_times_arr = jnp.asarray(obs_times)
    return DiracLatentMetadata(
        state_path_param_times=jnp.asarray([], dtype=obs_times_arr.dtype),
        state_path_param_coordinate_indices=jnp.asarray([], dtype=jnp.int32),
        free_flat_indices=jnp.asarray([], dtype=jnp.int32),
        state_shape=state_shape,
    )


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
