"""ODE forward-simulation backend."""

import jax
import jax.random as jr
from jax import Array

from dynestyx.inference.configs.simulator import ODESimulatorConfig
from dynestyx.models import DynamicalModel
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import _sample_initial_states, _tile_times
from dynestyx.solvers import solve_ode_state_path
from dynestyx.types import SimulatedResult
from dynestyx.utils import _build_control_path_eval


class ODESimulator(BaseSimulator):
    """Forward simulator for continuous-time deterministic dynamics (ODEs)."""

    def __init__(
        self,
        simulator_config: ODESimulatorConfig | None = None,
        *,
        n_simulations: int = 1,
    ):
        """Configure ODE integration.

        Args:
            simulator_config: Structured simulator settings. Defaults to
                `ODESimulatorConfig()` when omitted.
            n_simulations: Number of independent trajectories to simulate. State
                and observation paths have shape `(n_simulations, T, ...)`. Must
                be greater than or equal to one.
        """
        super().__init__(n_simulations=n_simulations)
        if simulator_config is None:
            simulator_config = ODESimulatorConfig()

        self.simulator_config = simulator_config
        self.diffeqsolve_settings = simulator_config.diffeqsolve_settings

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
            x_0=initial_state,
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
        """Run pure-JAX forward simulation for deterministic continuous-time models.

        Unlike :func:`dynestyx.simulate`, ``rng_key`` is consumed directly as an
        already-allocated simulation key and is not pre-split. Therefore,
        ``dynestyx.simulate(..., rng_key=root_key)`` is equivalent to
        ``ODESimulator.simulate(..., rng_key=jax.random.split(root_key)[1])``.
        """
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
            initial_state=initial_state,
            rng_key=rollout_key,
            times=times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
