"""Standalone latent-path scoring utilities."""

from __future__ import annotations

import dataclasses
import math

import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Real

from dynestyx.models import DynamicalModel
from dynestyx.observation_missingness import ObservationLogProb
from dynestyx.utils import (
    _get_dynamics_with_t0,
    _validate_control_dim,
    _validate_controls,
    _validate_site_sorting,
)


@dataclasses.dataclass
class DiscretePathScoreTerms:
    """Full discrete-time path-score decomposition."""

    init_log_prob: Array
    transition_log_probs: Array
    observation_log_probs: Array

    @property
    def latent_log_prob(self) -> Array:
        return self.init_log_prob + jnp.sum(self.transition_log_probs)

    @property
    def observation_log_prob(self) -> Array:
        return jnp.sum(self.observation_log_probs)

    @property
    def total_log_prob(self) -> Array:
        return self.latent_log_prob + self.observation_log_prob


def _chunked_sum(
    length: int,
    *,
    chunk_size: int | None,
    fn,
):
    """Sum scalar contributions either with one vmap or a scan of vmaps."""
    if length <= 0:
        return jnp.array(0.0)

    if chunk_size is None or chunk_size >= length:
        return jnp.sum(jax.vmap(fn)(jnp.arange(length)))

    n_chunks = math.ceil(length / chunk_size)
    padded = n_chunks * chunk_size
    idxs = jnp.arange(padded).reshape(n_chunks, chunk_size)
    valid = idxs < length
    safe_idxs = jnp.minimum(idxs, length - 1)
    zero = jnp.asarray(fn(0)) * 0.0

    def _scan_step(total, inputs):
        idx_chunk, valid_chunk = inputs
        vals = jax.vmap(fn)(idx_chunk)
        vals = jnp.where(valid_chunk, vals, zero)
        return total + jnp.sum(vals), None

    total, _ = lax.scan(_scan_step, zero, (safe_idxs, valid))
    return total


def observation_log_prob_terms(
    dynamics: DynamicalModel,
    latent_states: Array,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"],
    ctrl_values: Array | None = None,
    precomputed_filled_obs: Array | None = None,
    precomputed_obs_mask: Array | None = None,
) -> Array:
    """Return per-time observation log-probability contributions in JAX."""
    obs_times = jnp.asarray(obs_times)
    latent_states = jnp.asarray(latent_states)
    obs_values_arr = jnp.asarray(obs_values)
    obs_values_2d = (
        obs_values_arr[:, None] if obs_values_arr.ndim == 1 else obs_values_arr
    )
    log_prob = ObservationLogProb(
        dynamics=dynamics,
        obs_values=obs_values_2d,
        precomputed_filled_obs=precomputed_filled_obs,
        precomputed_obs_mask=precomputed_obs_mask,
    )

    def _observation_term(t_idx):
        u_t = None if ctrl_values is None else ctrl_values[t_idx]
        return log_prob.log_prob_step(
            x=latent_states[t_idx],
            u=u_t,
            t=obs_times[t_idx],
            t_idx=t_idx,
        )

    return jax.vmap(_observation_term)(jnp.arange(len(obs_times)))


def discrete_path_score_terms(
    dynamics: DynamicalModel,
    latent_states: Array,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None = None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    precomputed_filled_obs: Array | None = None,
    precomputed_obs_mask: Array | None = None,
) -> DiscretePathScoreTerms:
    """Return the discrete-time full-path score decomposition."""
    if dynamics.continuous_time:
        raise NotImplementedError(
            "discrete_path_score_terms currently supports discrete-time or "
            "discretized models only."
        )

    _validate_site_sorting(obs_times, name="obs_times")
    _validate_controls(obs_times, None, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)
    dynamics = _get_dynamics_with_t0(dynamics, obs_times, None)

    obs_times = jnp.asarray(obs_times)
    latent_states = jnp.asarray(latent_states)
    if obs_times.ndim != 1:
        raise ValueError(
            "discrete_path_score_terms currently expects obs_times with shape (time,)."
        )
    if latent_states.shape[0] != obs_times.shape[0]:
        raise ValueError(
            "latent_states and obs_times must have the same number of time steps."
        )

    ctrl_values = None if ctrl_values is None else jnp.asarray(ctrl_values)
    init_log_prob = dynamics.initial_condition.log_prob(latent_states[0])

    def _transition_term(t_idx):
        u_now = None if ctrl_values is None else ctrl_values[t_idx]
        transition_dist = dynamics.state_evolution(
            x=latent_states[t_idx],
            u=u_now,
            t_now=obs_times[t_idx],
            t_next=obs_times[t_idx + 1],
        )
        return transition_dist.log_prob(latent_states[t_idx + 1])

    transition_log_probs = (
        jax.vmap(_transition_term)(jnp.arange(len(obs_times) - 1))
        if len(obs_times) > 1
        else jnp.zeros((0,), dtype=init_log_prob.dtype)
    )

    if obs_values is None:
        observation_log_probs = jnp.zeros((len(obs_times),), dtype=init_log_prob.dtype)
    else:
        observation_log_probs = observation_log_prob_terms(
            dynamics,
            latent_states,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_values=ctrl_values,
            precomputed_filled_obs=precomputed_filled_obs,
            precomputed_obs_mask=precomputed_obs_mask,
        )

    return DiscretePathScoreTerms(
        init_log_prob=init_log_prob,
        transition_log_probs=transition_log_probs,
        observation_log_probs=observation_log_probs,
    )


def path_log_prob(
    dynamics: DynamicalModel,
    latent_states: Array,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None = None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    chunk_size: int | None = None,
) -> jax.Array:
    """Return the full latent-path score for a discrete-time model.

    This computes

    ``log p(x_0) + sum_k log p(x_{k+1} | x_k, u_k) + sum_k log p(y_k | x_k, u_k)``

    while respecting the repository's existing missing-observation machinery for
    the observation terms.
    """
    terms = discrete_path_score_terms(
        dynamics,
        latent_states,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    init_log_prob = terms.init_log_prob

    transition_log_prob = _chunked_sum(
        len(terms.transition_log_probs),
        chunk_size=chunk_size,
        fn=lambda t_idx: terms.transition_log_probs[t_idx],
    )
    observation_log_prob = _chunked_sum(
        len(terms.observation_log_probs),
        chunk_size=chunk_size,
        fn=lambda t_idx: terms.observation_log_probs[t_idx],
    )

    return init_log_prob + transition_log_prob + observation_log_prob


__all__ = [
    "DiscretePathScoreTerms",
    "discrete_path_score_terms",
    "observation_log_prob_terms",
    "path_log_prob",
]
