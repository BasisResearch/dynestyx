"""ODE forward-simulation backend."""

from typing import cast

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from dynestyx.inference.configs.simulator import ODESimulatorConfig
from dynestyx.models import DynamicalModel
from dynestyx.simulation.base import (
    _SIMULATOR_CONFIG_UNSET,
    BaseSimulator,
    _sample_initial_states,
    _tile_times,
    _validate_no_config_and_direct_kwargs,
)
from dynestyx.solvers import solve_ode_state_path
from dynestyx.types import SimulatedResult
from dynestyx.utils import _build_control_path_eval


class ODESimulator(BaseSimulator):
    """Forward simulator for continuous-time deterministic dynamics (ODEs)."""

    def __init__(
        self,
        simulator_config: ODESimulatorConfig | None = None,
        *,
        solver: dfx.AbstractSolver | object = _SIMULATOR_CONFIG_UNSET,
        adjoint: dfx.AbstractAdjoint | object = _SIMULATOR_CONFIG_UNSET,
        stepsize_controller: dfx.AbstractStepSizeController
        | object = _SIMULATOR_CONFIG_UNSET,
        dt0: float | int | Array | object = _SIMULATOR_CONFIG_UNSET,
        max_steps: int | object = _SIMULATOR_CONFIG_UNSET,
        n_simulations: int = 1,
    ):
        _validate_no_config_and_direct_kwargs(
            simulator_config=simulator_config,
            config_name="simulator_config",
            direct_kwargs={
                "solver": solver,
                "adjoint": adjoint,
                "stepsize_controller": stepsize_controller,
                "dt0": dt0,
                "max_steps": max_steps,
            },
        )

        if simulator_config is None:
            solver_value: dfx.AbstractSolver = cast(
                dfx.AbstractSolver,
                dfx.Tsit5() if solver is _SIMULATOR_CONFIG_UNSET else solver,
            )
            adjoint_value: dfx.AbstractAdjoint = cast(
                dfx.AbstractAdjoint,
                (
                    dfx.RecursiveCheckpointAdjoint()
                    if adjoint is _SIMULATOR_CONFIG_UNSET
                    else adjoint
                ),
            )
            stepsize_controller_value: dfx.AbstractStepSizeController = cast(
                dfx.AbstractStepSizeController,
                (
                    dfx.ConstantStepSize()
                    if stepsize_controller is _SIMULATOR_CONFIG_UNSET
                    else stepsize_controller
                ),
            )
            dt0_value: float | int | Array
            if dt0 is _SIMULATOR_CONFIG_UNSET:
                dt0_value = 1e-3
            else:
                dt0_value = cast(float | int | Array, dt0)
            max_steps_value: int
            if max_steps is _SIMULATOR_CONFIG_UNSET:
                max_steps_value = 100_000
            else:
                max_steps_value = cast(int, max_steps)
            simulator_config = ODESimulatorConfig(
                solver=solver_value,
                adjoint=adjoint_value,
                stepsize_controller=stepsize_controller_value,
                dt0=dt0_value,
                max_steps=max_steps_value,
            )

        self.simulator_config = simulator_config
        self.diffeqsolve_settings = simulator_config.diffeqsolve_settings()
        self.n_simulations = n_simulations

    def _simulate_forward_from_initial_state(
        self,
        dynamics: DynamicalModel,
        *,
        initial_state: Array,
        rng_key: Array,
        times: Array,
        ctrl_times=None,
        ctrl_values=None,
    ) -> SimulatedResult:
        """Run pure forward simulation for a deterministic continuous-time model."""
        n_sim = initial_state.shape[0]

        control_path_eval = _build_control_path_eval(ctrl_times, ctrl_values, times)

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]
        obs_keys = jr.split(rng_key, n_sim)

        def _sim_one_trajectory(x0: Array, *, obs_key: Array) -> tuple[Array, Array]:
            states = solve_ode_state_path(
                dynamics,
                t0=t0,
                initial_state=x0,
                path_times=times,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                diffeqsolve_settings=self.diffeqsolve_settings,
            )
            observations = self._emit_observations(
                "",
                dynamics,
                states,
                times,
                None,
                control_path_eval,
                key=obs_key,
            )
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(
            initial_state, obs_key=obs_keys
        )
        return SimulatedResult(
            times=_tile_times(times, n_sim),
            initial_states=jnp.asarray(initial_state),
            states=states,
            observations=observations,
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
        """Run pure-JAX forward simulation for deterministic continuous-time models."""
        times = obs_times if obs_times is not None else predict_times
        if times is None:
            raise ValueError("obs_times or predict_times must be provided")

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
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
