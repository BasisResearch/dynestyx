"""Discrete-time forward-simulation backend."""

import dataclasses
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from dynestyx.models import DynamicalModel
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.simulation.base import (
    BaseSimulator,
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

    times_arr = jnp.asarray(times)
    ctrl_times_arr = jnp.asarray(ctrl_times)
    ctrl_values_arr = jnp.asarray(ctrl_values)
    idx = jnp.searchsorted(ctrl_times_arr, times_arr, side="left")
    max_idx = ctrl_times_arr.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    matched = (idx < ctrl_times_arr.shape[0]) & (ctrl_times_arr[safe_idx] == times_arr)
    _raise_now_or_error_if(
        times_arr,
        jnp.any(~matched),
        "ctrl_times must contain every discrete simulation time exactly.",
    )
    return ctrl_values_arr[safe_idx]


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
        state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)
        n_sim = initial_state.shape[0]
        T = len(times)
        sim_keys = jr.split(rng_key, n_sim)
        ctrl_eval = (
            (lambda t: ctrl_values[jnp.searchsorted(times, t, side="left")])
            if ctrl_values is not None
            else None
        )

        def _step_dists(x_prev, t_idx):
            t_now = times[t_idx]
            t_next = times[t_idx + 1]
            u_now = _get_val_or_None(ctrl_values, t_idx)
            u_next = _get_val_or_None(ctrl_values, t_idx + 1)
            trans_dist = state_transition(x=x_prev, u=u_now, t_now=t_now, t_next=t_next)
            return t_next, u_next, trans_dist

        def _sim_one_trajectory(key: Array, x0: Array) -> tuple[Array, Array]:
            if T == 1:
                states = jnp.expand_dims(x0, axis=0)
                observations = self._emit_observations(
                    "",
                    dynamics,
                    states,
                    times,
                    None,
                    ctrl_eval,
                    key=key,
                )
                return states, observations

            key_trans, key_obs = jr.split(key)

            def _step(carry, t_idx):
                x_prev, key_curr = carry
                key_next, k_trans = jr.split(key_curr, 2)
                _, _, trans_dist = _step_dists(x_prev, t_idx)
                x_t = trans_dist.sample(k_trans)
                return (x_t, key_next), x_t

            (_, _), scan_states = jax.lax.scan(
                _step, (x0, key_trans), jnp.arange(T - 1)
            )
            states = jnp.concatenate([jnp.expand_dims(x0, 0), scan_states], axis=0)
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
            initial_states=jnp.asarray(initial_state),
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
            times=jnp.asarray(times),
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
            initial_state=jnp.asarray(initial_state),
            rng_key=rollout_key,
            times=times,
            ctrl_values=aligned_ctrl_values,
        )
