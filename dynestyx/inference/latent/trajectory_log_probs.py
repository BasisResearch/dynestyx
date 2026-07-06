"""Pure-JAX joint trajectory log-probability helpers."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from dynestyx.inference.latent.state_path import assemble_state_path
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.observation_missingness import ObservationLogProb
from dynestyx.utils import (
    _get_val_or_None,
    _raise_now_or_error_if,
)


@dataclasses.dataclass
class TrajectoryLogProbTerms:
    """Pure-JAX decomposition of ``log p(x, y | ...)``."""

    initial_log_prob: Array
    transition_log_probs: Array
    observation_log_probs: Array

    @property
    def joint_log_prob(self) -> Array:
        return (
            self.initial_log_prob
            + jnp.sum(self.transition_log_probs)
            + jnp.sum(self.observation_log_probs)
        )


def _scan_chunked_vmap(
    fn: Callable[[Array], Array],
    size: int,
    *,
    chunk_size: int | None,
    dtype,
) -> Array:
    """Evaluate a scalar per-index function via a scan of vmapped chunks."""
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
    """Gather values defined on ``source_times`` at ``query_times`` exactly."""
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
    obs_values: Array,
    *,
    obs_values_filled: Array | None,
    obs_mask: Array | None,
) -> ObservationLogProb:
    """Construct the missingness-aware observation scorer."""
    obs_values_for_helper = obs_values[:, None] if obs_values.ndim == 1 else obs_values
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
    return ObservationLogProb(
        dynamics=dynamics,
        obs_values=obs_values_for_helper,
        precomputed_filled_obs=filled_obs_for_helper,
        precomputed_obs_mask=obs_mask_for_helper,
    )


def _control_values_at_times(
    ctrl_times: Array | None,
    ctrl_values: Array | None,
    query_times: Array | None,
) -> Array | None:
    """Return control values aligned exactly to ``query_times`` when present."""
    if ctrl_times is None or ctrl_values is None or query_times is None:
        return None
    return _gather_by_exact_time(
        jnp.asarray(ctrl_values),
        jnp.asarray(ctrl_times),
        jnp.asarray(query_times),
        value_name="ctrl_values",
    )


def _compute_log_prob_terms_from_state_trajectory(
    dynamics: DynamicalModel,
    *,
    state_path: Array,
    state_path_times: Array,
    obs_times: Array | None = None,
    obs_values: Array | None = None,
    obs_values_filled: Array | None = None,
    obs_mask: Array | None = None,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    chunk_size: int | None = None,
    observations_are_exact_constraints: bool = False,
) -> TrajectoryLogProbTerms:
    """Compute ``log p(x, y | ...)`` from full reconstructed state values.

    This helper assumes the caller has already converted any latent
    parameterization into a concrete state path on a concrete state-time grid.
    In particular, ``state_path`` is the actual latent trajectory

    ``x = (x_0, x_1, ..., x_T)``

    used in the
    initial-condition term, the transition terms, and the observation terms.

    This differs from ``state_path_params``, which may be a compressed or
    model-specific parameterization ``z`` with ``x = g(z)``:
    - discrete / discretized v1: full latent path, often equal to
      ``state_path``,
    - deterministic continuous-time models: initial condition only,
    - exact-observation compressed layouts: only the free coordinates.
    """
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

    observation_log_probs = jnp.zeros((0,), dtype=initial_log_prob.dtype)
    if (
        not observations_are_exact_constraints
        and obs_times is not None
        and obs_values is not None
    ):
        obs_times_arr = jnp.asarray(obs_times)
        obs_states = _gather_by_exact_time(
            state_path,
            state_path_times,
            obs_times_arr,
            value_name="state_path",
        )
        obs_ctrl_values = _control_values_at_times(
            ctrl_times, ctrl_values, obs_times_arr
        )
        observation_scorer = _prepare_observation_log_prob(
            dynamics,
            jnp.asarray(obs_values),
            obs_values_filled=obs_values_filled,
            obs_mask=obs_mask,
        )

        def _obs_at(i: Array) -> Array:
            u_t = _get_val_or_None(obs_ctrl_values, i)
            return observation_scorer.log_prob_step(
                x=obs_states[i],
                u=u_t,
                t=obs_times_arr[i],
                t_idx=i,
            )

        observation_log_probs = _scan_chunked_vmap(
            _obs_at,
            obs_times_arr.shape[0],
            chunk_size=chunk_size,
            dtype=initial_log_prob.dtype,
        )

    return TrajectoryLogProbTerms(
        initial_log_prob=initial_log_prob,
        transition_log_probs=transition_log_probs,
        observation_log_probs=observation_log_probs,
    )


def compute_trajectory_log_prob_terms(
    dynamics: DynamicalModel,
    *,
    state_path_params: Array,
    state_path_param_times: Array,
    obs_times: Array | None = None,
    obs_values: Array | None = None,
    obs_values_filled: Array | None = None,
    obs_mask: Array | None = None,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    chunk_size: int | None = None,
    ode_diffeqsolve_settings: dict[str, Any] | None = None,
) -> TrajectoryLogProbTerms:
    """Compute ``log p(x, y | ...)`` from a state-path parameterization.

    Let ``z = state_path_params`` denote the free variables supplied by the
    caller, and let ``x = state_path`` denote the full trajectory appearing in
    the model. This function evaluates ``log p(x, y | ...)`` after first
    reconstructing ``x = g(z)``.

    The path parameters are not required to equal the full trajectory that appears in the
    probabilistic model:
    - discrete / discretized v1: ``state_path_params`` are the full latent path,
    - deterministic continuous-time models: ``state_path_params`` are only the
      initial condition and the ODE solution reconstructs the rest,
    - compressed exact-observation layouts are handled by a different assembly
      path before calling the internal assembled-state scorer.

    This function therefore has two stages:
    1. assemble ``state_path_params`` into a full ``state_path`` on ``state_path_times``
    2. score that assembled trajectory
    """
    state_path_param_times = jnp.asarray(state_path_param_times)
    _raise_now_or_error_if(
        state_path_param_times,
        state_path_param_times.shape[0] < 1,
        "state_path_param_times must contain at least one time point.",
    )

    if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
        raise ValueError(
            "dsx.log_prob does not yet support native SDE models. "
            "Please discretize the model first."
        )

    assembled = assemble_state_path(
        dynamics,
        state_path_params=state_path_params,
        state_path_param_times=state_path_param_times,
        obs_times=obs_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        ode_diffeqsolve_settings=ode_diffeqsolve_settings,
    )

    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if state_path_param_times.shape[0] != 1:
            raise ValueError(
                "Deterministic continuous-time models expect exactly one latent "
                "path parameter in dsx.log_prob: the initial condition."
            )

    return _compute_log_prob_terms_from_state_trajectory(
        dynamics,
        state_path=assembled.state_path,
        state_path_times=assembled.state_path_times,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        chunk_size=chunk_size,
    )
