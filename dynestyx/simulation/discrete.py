"""Discrete-time forward-simulation backend."""

import dataclasses
from typing import cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from jax import Array

from dynestyx.models import DynamicalModel
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.simulation.base import (
    BaseSimulator,
    _ensure_trailing_dim,
    _simulated_result_to_dict,
    _tile_times,
)
from dynestyx.types import SimulatedResult
from dynestyx.utils import _get_val_or_None


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

        def _step_dists(x_prev, t_idx):
            t_now = times[t_idx]
            t_next = times[t_idx + 1]
            u_now = _get_val_or_None(ctrl_values, t_idx)
            u_next = _get_val_or_None(ctrl_values, t_idx + 1)
            trans_dist = state_transition(x=x_prev, u=u_now, t_now=t_now, t_next=t_next)
            return t_next, u_next, trans_dist

        def _sim_one_trajectory(key: Array, x0: Array) -> tuple[Array, Array]:
            key, y0_key = jr.split(key)
            u_0 = _get_val_or_None(ctrl_values, 0)
            y_0 = dynamics.observation_model(x=x0, u=u_0, t=times[0]).sample(y0_key)

            if T == 1:
                states = jnp.expand_dims(x0, axis=0)
                observations = jnp.expand_dims(y_0, axis=0)
                return states, observations

            def _step(carry, t_idx):
                x_prev, key_curr = carry
                key_next, k_trans, k_obs = jr.split(key_curr, 3)
                t_next, u_next, trans_dist = _step_dists(x_prev, t_idx)
                x_t = trans_dist.sample(k_trans)
                y_t = dynamics.observation_model(x=x_t, u=u_next, t=t_next).sample(
                    k_obs
                )
                return (x_t, key_next), (x_t, y_t)

            (_, _), (scan_states, scan_obs) = jax.lax.scan(
                _step, (x0, key), jnp.arange(T - 1)
            )
            states = jnp.concatenate([jnp.expand_dims(x0, 0), scan_states], axis=0)
            observations = jnp.concatenate([jnp.expand_dims(y_0, 0), scan_obs], axis=0)
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(sim_keys, initial_state)
        return SimulatedResult(
            times=_tile_times(times, n_sim),
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

        initial_key, rollout_key = jr.split(rng_key)
        initial_state = dynamics.initial_condition.sample(
            initial_key, sample_shape=(self.n_simulations,)
        )
        return self._simulate_forward_from_initial_state(
            dynamics,
            initial_state=jnp.asarray(initial_state),
            rng_key=rollout_key,
            times=times,
            ctrl_values=ctrl_values,
        )

    def _simulate(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        _obs_values_filled=None,
        _obs_mask=None,
        _obs_has_missing=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, Array]:
        """Unroll a discrete-time model as a NumPyro forward simulator."""
        if obs_times is not None or obs_values is not None:
            raise ValueError(
                "DiscreteTimeSimulator is generation-only. Use predict_times for "
                "simulation, or LatentPathBuilder / Filter for inference with observations."
            )

        times = predict_times
        if times is None:
            raise ValueError("predict_times must be provided")
        if len(times) < 1:
            raise ValueError("predict_times must contain at least one timepoint")

        with numpyro.plate(f"{name}_n_simulations", self.n_simulations):
            initial_state = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        prng_key = numpyro.prng_key()
        if prng_key is None:
            raise ValueError("PRNG key required for simulation")
        result = self._simulate_forward_from_initial_state(
            dynamics,
            initial_state=jnp.asarray(initial_state),
            rng_key=prng_key,
            times=times,
            ctrl_values=ctrl_values,
        )
        return _simulated_result_to_dict(result)
