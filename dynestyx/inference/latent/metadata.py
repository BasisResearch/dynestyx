"""Latent-path metadata, indexing, and canonicalization helpers."""

from __future__ import annotations

import dataclasses

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


@dataclasses.dataclass
class DiracLatentMetadata:
    """Concrete indexing metadata for exact-observation latent compression."""

    state_path_param_times: Array
    state_path_param_coordinate_indices: Array
    free_flat_indices: Array
    state_shape: tuple[int, ...]


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


__all__ = [
    "DiracLatentMetadata",
    "canonicalize_dirac_state_path_params",
    "canonicalize_state_path_params",
    "fully_observed_dirac_state_path_param_metadata",
    "infer_dirac_state_path_param_metadata",
    "infer_state_path_param_times",
    "prepare_dirac_state_path_metadata",
]
