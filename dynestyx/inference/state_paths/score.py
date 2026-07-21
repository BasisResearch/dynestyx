"""Pure-JAX state-path scoring helpers."""

from __future__ import annotations

import math
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from dynestyx.models import DeterministicContinuousTimeStateEvolution, DynamicalModel
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
    prepare_observation_log_prob,
)
from dynestyx.utils import _get_val_or_None, _raise_now_or_error_if


def _scan_chunked_vmap(
    fn: Callable[[Array], Array],
    size: int,
    *,
    chunk_size: int | None,
    dtype,
) -> Array:
    if size == 0:
        return jnp.zeros((0,), dtype=dtype)

    if chunk_size is None or chunk_size <= 0 or chunk_size >= size:
        return jax.vmap(fn)(jnp.arange(size))

    n_chunks = math.ceil(size / chunk_size)
    padded_size = n_chunks * chunk_size
    chunked_indices = jnp.arange(padded_size).reshape(n_chunks, chunk_size)

    def _chunk_step(carry, idx_chunk):
        safe_idx_chunk = jnp.minimum(idx_chunk, size - 1)
        chunk_values = jax.vmap(fn)(safe_idx_chunk)
        valid_mask = idx_chunk < size
        masked_chunk_values = jnp.where(
            valid_mask, chunk_values, jnp.zeros_like(chunk_values)
        )
        return carry, masked_chunk_values

    _, chunk_outputs = lax.scan(_chunk_step, None, chunked_indices)
    return chunk_outputs.reshape(padded_size)[:size]


def _gather_by_exact_time(
    values: Array,
    source_times: Array,
    query_times: Array,
    *,
    value_name: str,
) -> Array:
    source = jnp.asarray(source_times)
    query = jnp.asarray(query_times)
    if query.size == 0:
        return values[:0]

    idx = jnp.searchsorted(source, query, side="left")
    max_idx = source.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    matched = (idx < source.shape[0]) & (source[safe_idx] == query)
    _ = eqx.error_if(
        query,
        jnp.any(~matched),
        f"{value_name} must be defined exactly at every requested query time.",
    )
    return values[safe_idx]


def _prepare_observation_log_prob(
    dynamics: DynamicalModel,
    obs_times: Array,
    obs_values: Array,
    *,
    obs_values_filled: Array | None,
    obs_mask: Array | None,
    missing_observation_strategy: MissingObservationStrategy,
    missing_obs_values: Array | None,
    missing_obs_metadata: MissingObservationMetadata | None,
):
    obs_values_for_helper = obs_values[:, None] if obs_values.ndim == 1 else obs_values
    obs_times_for_helper = jnp.asarray(obs_times)
    filled_obs_for_helper = (
        None
        if obs_values_filled is None
        else (obs_values_filled[:, None] if obs_values.ndim == 1 else obs_values_filled)
    )
    obs_mask_for_helper = (
        None
        if obs_mask is None
        else (obs_mask[:, None] if obs_values.ndim == 1 else obs_mask)
    )
    missing_obs_values_for_helper = (
        None
        if missing_obs_values is None
        else (
            missing_obs_values[:, None]
            if obs_values.ndim == 1 and jnp.asarray(missing_obs_values).ndim == 1
            else missing_obs_values
        )
    )
    return prepare_observation_log_prob(
        dynamics=dynamics,
        obs_values=obs_values_for_helper,
        obs_times=obs_times_for_helper,
        precomputed_filled_obs=filled_obs_for_helper,
        precomputed_obs_mask=obs_mask_for_helper,
        missing_observation_strategy=missing_observation_strategy,
        missing_obs_values=missing_obs_values_for_helper,
        missing_obs_metadata=missing_obs_metadata,
    )


def _control_values_at_times(
    ctrl_times: Array | None,
    ctrl_values: Array | None,
    query_times: Array | None,
) -> Array | None:
    if ctrl_times is None or ctrl_values is None or query_times is None:
        return None
    return _gather_by_exact_time(
        jnp.asarray(ctrl_values),
        jnp.asarray(ctrl_times),
        jnp.asarray(query_times),
        value_name="ctrl_values",
    )


def compute_state_path_log_prob(
    dynamics: DynamicalModel,
    *,
    state_path: Array,
    state_path_times: Array,
    obs_times: Array | None = None,
    obs_values: Array | None = None,
    obs_values_filled: Array | None = None,
    obs_mask: Array | None = None,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    missing_obs_values: Array | None = None,
    missing_obs_metadata: MissingObservationMetadata | None = None,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    chunk_size: int | None = None,
    observations_are_exact_constraints: bool = False,
) -> Array:
    state_path_times = jnp.asarray(state_path_times)
    state_path = jnp.asarray(state_path)
    _raise_now_or_error_if(
        state_path_times,
        state_path_times.shape[0] < 1,
        "state_path_times must contain at least one time point.",
    )

    initial_log_prob = dynamics.initial_condition.log_prob(state_path[0])

    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        transition_log_probs = jnp.zeros((0,), dtype=initial_log_prob.dtype)
    else:
        state_ctrl_values = _control_values_at_times(
            ctrl_times, ctrl_values, state_path_times
        )
        n_transitions = max(state_path_times.shape[0] - 1, 0)
        if n_transitions == 0:
            transition_log_probs = jnp.zeros((0,), dtype=initial_log_prob.dtype)
        else:

            def _transition_at(i: Array) -> Array:
                x_prev = state_path[i]
                x_next = state_path[i + 1]
                u_prev = _get_val_or_None(state_ctrl_values, i)
                transition_dist = dynamics.state_evolution(
                    x=x_prev,
                    u=u_prev,
                    t_now=state_path_times[i],
                    t_next=state_path_times[i + 1],
                )
                return transition_dist.log_prob(x_next)

            transition_log_probs = _scan_chunked_vmap(
                _transition_at,
                n_transitions,
                chunk_size=chunk_size,
                dtype=initial_log_prob.dtype,
            )

    if obs_times is None or obs_values is None:
        return initial_log_prob + jnp.sum(transition_log_probs)

    state_at_obs_times = _gather_by_exact_time(
        state_path,
        state_path_times,
        jnp.asarray(obs_times),
        value_name="state_path",
    )
    if observations_are_exact_constraints:
        return initial_log_prob + jnp.sum(transition_log_probs)

    obs_ctrl_values = _control_values_at_times(ctrl_times, ctrl_values, obs_times)
    observation_log_prob, _, _, _ = _prepare_observation_log_prob(
        dynamics,
        jnp.asarray(obs_times),
        jnp.asarray(obs_values),
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        missing_observation_strategy=missing_observation_strategy,
        missing_obs_values=missing_obs_values,
        missing_obs_metadata=missing_obs_metadata,
    )

    def _observation_at(i: Array) -> Array:
        return observation_log_prob(
            x=state_at_obs_times[i],
            u=_get_val_or_None(obs_ctrl_values, i),
            t=jnp.asarray(obs_times)[i],
            t_idx=i,
        )

    observation_log_probs = _scan_chunked_vmap(
        _observation_at,
        jnp.asarray(obs_times).shape[0],
        chunk_size=chunk_size,
        dtype=initial_log_prob.dtype,
    )

    return (
        initial_log_prob
        + jnp.sum(transition_log_probs)
        + jnp.sum(observation_log_probs)
    )


__all__ = [
    "compute_state_path_log_prob",
]
