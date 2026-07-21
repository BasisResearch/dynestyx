"""SDE simulator backend."""

import jax
import jax.random as jr
from jax import Array

from dynestyx.inference.configs.simulator import SDESimulatorConfig
from dynestyx.models import DynamicalModel, StochasticContinuousTimeStateEvolution
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import _sample_initial_states, _tile_times
from dynestyx.solvers import solve_sde_state_path
from dynestyx.types import SimulatedResult
from dynestyx.utils import _build_control_path_eval


class SDESimulator(BaseSimulator):
    """Simulator for continuous-time stochastic dynamics (SDEs).

    This simulator integrates a `ContinuousTimeStateEvolution` with nonzero diffusion

    Controls:
        If `ctrl_times` / `ctrl_values` are provided at the `dsx.sample(...)` site,
        controls are interpolated with a right-continuous rectilinear rule
        (`left=False`), i.e., the control at time `t_k` is `ctrl_values[k]`.

    Deterministic outputs:
        When run through `dsx.sample(...)`, the simulator records `"x_0"`,
        `"times"`, `"states"`, and `"observations"` as
        `numpyro.deterministic(...)` sites.

    Important:
        - Conditioning on `obs_values` with an SDE unroller typically yields a
          very high-dimensional latent path and is usually a **poor inference
          strategy** for parameters. Prefer filtering (`Filter` with
          `ContinuousTime*Config`) or particle methods instead.

    Tip for speed:
        - Use `SDESimulatorConfig(source="em_scan")` if you are happy with a simple Euler-Maruyama forward simulation
          (10–20x faster than Diffrax's implementation; see
          [Diffrax Issue #517](https://github.com/patrick-kidger/diffrax/issues/517)).
        - Use `SDESimulatorConfig(source="diffrax")` if you want greater flexibility in the solver and step-size control.
    """

    def __init__(
        self,
        simulator_config: SDESimulatorConfig | None = None,
        *,
        n_simulations: int = 1,
    ):
        """Configure SDE integration settings.

        Args:
            simulator_config: Structured simulator settings. Defaults to
                `SDESimulatorConfig()` when omitted.
            n_simulations: Number of independent trajectory simulations. When > 1,
                states and observations have an extra leading dimension (n_simulations, T, ...).
        """
        if simulator_config is None:
            simulator_config = SDESimulatorConfig()

        self.simulator_config = simulator_config
        self.diffeqsolve_settings = simulator_config.diffeqsolve_settings
        self.n_simulations = n_simulations
        self.source = simulator_config.source
        self.tol_vbt = simulator_config.resolved_tol_vbt

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
        """Run pure forward SDE simulation from provided initial states."""
        n_sim = initial_state.shape[0]

        control_path_eval = _build_control_path_eval(ctrl_times, ctrl_values, times)

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]
        sim_keys = jr.split(rng_key, n_sim)

        def _sim_one_trajectory(key: Array, x0: Array) -> tuple[Array, Array]:
            k_solve, k_obs = jr.split(key, 2)
            states = solve_sde_state_path(
                source=self.source,
                dynamics=dynamics,
                t0=t0,
                path_times=times,
                initial_state=x0,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                diffeqsolve_settings=self.diffeqsolve_settings,
                key=k_solve,
                tol_vbt=self.tol_vbt,
            )
            observations = self._emit_observations(
                "",
                dynamics,
                states,
                times,
                None,
                control_path_eval,
                key=k_obs,
            )
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(sim_keys, initial_state)
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
        """Run pure-JAX forward simulation for stochastic continuous-time models."""
        if not isinstance(
            dynamics.state_evolution, StochasticContinuousTimeStateEvolution
        ):
            raise NotImplementedError(
                "SDESimulator only works with StochasticContinuousTimeStateEvolution, got "
                f"{type(dynamics.state_evolution)}"
            )
        if obs_times is not None:
            raise ValueError(
                "obs_times must not be provided to an SDESimulator; use predict_times for "
                "forward simulation, or use a filter / discretization workflow for inference."
            )

        times = predict_times
        if times is None:
            raise ValueError("predict_times must be provided for SDESimulator.")

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
