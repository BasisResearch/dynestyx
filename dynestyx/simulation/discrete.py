"""Discrete-time forward-simulation backend."""

import dataclasses
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from dynestyx.models import DynamicalModel
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import (
    _ensure_trailing_dim,
    _sample_initial_states,
    _tile_times,
)
from dynestyx.types import SimulatedResult
from dynestyx.utils import _get_val_or_None, _raise_now_or_error_if


def _align_ctrl_values_to_times(
    *,
    times: Array,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
) -> Array | None:
    """Return control values aligned to the simulator time grid."""
    if ctrl_times is None or ctrl_values is None:
        return ctrl_values

    idx = jnp.searchsorted(ctrl_times, times, side="left")
    max_idx = ctrl_times.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    matched = (idx < ctrl_times.shape[0]) & (ctrl_times[safe_idx] == times)
    _raise_now_or_error_if(
        times,
        jnp.any(~matched),
        "ctrl_times must contain every discrete simulation time exactly.",
    )
    return ctrl_values[safe_idx]


def _sample_discrete_state_path_from_initial_state(
    dynamics: DynamicalModel,
    *,
    initial_state: Array,
    rng_key: Array,
    times: Array,
    ctrl_values: Array | None,
) -> Array:
    """Sample one canonical discrete state path from a fixed initial state."""
    if len(times) == 1:
        return jnp.expand_dims(initial_state, axis=0)

    state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)

    def _step(carry, t_idx):
        x_prev, key_curr = carry
        key_next, key_transition = jr.split(key_curr)
        transition_dist = state_transition(
            x=x_prev,
            u=_get_val_or_None(ctrl_values, t_idx),
            t_now=times[t_idx],
            t_next=times[t_idx + 1],
        )
        x_t = transition_dist.sample(key_transition)
        return (x_t, key_next), x_t

    (_, _), scan_states = jax.lax.scan(
        _step,
        (initial_state, rng_key),
        jnp.arange(len(times) - 1),
    )
    return jnp.concatenate([jnp.expand_dims(initial_state, 0), scan_states], axis=0)


def _sample_discrete_state_path(
    rng_key: Array,
    *,
    dynamics: DynamicalModel,
    times: Array,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
) -> Array:
    """Sample one state path from the discrete dynamical prior."""
    aligned_ctrl_values = _align_ctrl_values_to_times(
        times=times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    initial_key, transition_key = jr.split(rng_key)
    initial_state = dynamics.initial_condition.sample(initial_key)
    return _sample_discrete_state_path_from_initial_state(
        dynamics,
        initial_state=initial_state,
        rng_key=transition_key,
        times=times,
        ctrl_values=aligned_ctrl_values,
    )


@dataclasses.dataclass
class DiscreteTimeSimulator(BaseSimulator):
    """Forward simulator for discrete-time dynamical models."""

    n_simulations: int = 1

    def _simulate_forward_from_initial_state(
        self,
        dynamics: DynamicalModel,
        *,
        initial_state: Array,
        rng_key: Array,
        times: Array,
        ctrl_values: Array | None,
    ) -> SimulatedResult:
        """Run pure forward simulation for a discrete-time model."""
        n_sim = initial_state.shape[0]
        sim_keys = jr.split(rng_key, n_sim)
        ctrl_eval = (
            (lambda t: ctrl_values[jnp.searchsorted(times, t, side="left")])
            if ctrl_values is not None
            else None
        )

        def _sim_one_trajectory(key: Array, x0: Array) -> tuple[Array, Array]:
            key_states, key_obs = jr.split(key)
            states = _sample_discrete_state_path_from_initial_state(
                dynamics,
                initial_state=x0,
                rng_key=key_states,
                times=times,
                ctrl_values=ctrl_values,
            )
            observations = self._emit_observations(
                "",
                dynamics,
                states,
                times,
                None,
                ctrl_eval,
                key=key_obs,
            )
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(sim_keys, initial_state)
        return SimulatedResult(
            times=_tile_times(times, n_sim),
            x_0=initial_state,
            states=_ensure_trailing_dim(states),
            observations=_ensure_trailing_dim(observations),
        )

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: Array,
        obs_times=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> SimulatedResult:
        """Run pure-JAX forward simulation for a discrete-time model."""
        times = obs_times if obs_times is not None else predict_times
        if times is None:
            raise ValueError("obs_times or predict_times must be provided")

        aligned_ctrl_values = _align_ctrl_values_to_times(
            times=times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
        initial_key, rollout_key = jr.split(rng_key)
        initial_state = _sample_initial_states(
            dynamics.initial_condition,
            rng_key=initial_key,
            n_simulations=self.n_simulations,
        )
        return self._simulate_forward_from_initial_state(
            dynamics,
            initial_state=initial_state,
            rng_key=rollout_key,
            times=times,
            ctrl_values=aligned_ctrl_values,
        )
