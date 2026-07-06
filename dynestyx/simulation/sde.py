"""SDE simulator backend."""

from collections.abc import Callable
from typing import Literal, cast

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from jax import Array

from dynestyx.inference.configs.simulator import SDESimulatorConfig
from dynestyx.models import DynamicalModel, StochasticContinuousTimeStateEvolution
from dynestyx.simulation.base import (
    _SIMULATOR_CONFIG_UNSET,
    BaseSimulator,
    _emit_observations,
    _simulated_result_to_dict,
    _tile_times,
    _validate_no_config_and_direct_kwargs,
)
from dynestyx.solvers import solve_sde
from dynestyx.types import SimulatedResult
from dynestyx.utils import _build_control_path


class SDESimulator(BaseSimulator):
    """Simulator for continuous-time stochastic dynamics (SDEs).

    This simulator integrates a `ContinuousTimeStateEvolution` with nonzero diffusion
    using Diffrax and a `VirtualBrownianTree` (see the Diffrax docs on
    [Brownian controls](https://docs.kidger.site/diffrax/api/brownian/)). It constructs a NumPyro generative
    model with state sample sites (starting at `"x_0"`) and observation sample sites
    (`"y_0"`, `"y_1"`, ...).

    Controls:
        If `ctrl_times` / `ctrl_values` are provided at the `dsx.sample(...)` site,
        controls are interpolated with a right-continuous rectilinear rule
        (`left=False`), i.e., the control at time `t_k` is `ctrl_values[k]`.

    Deterministic outputs:
        When run, the simulator records `"times"`, `"states"`, and `"observations"`
        as `numpyro.deterministic(...)` sites.

    Important:
        - This is intended for **simulation / predictive checks** inside NumPyro.
        - Conditioning on `obs_values` with an SDE unroller typically yields a
          very high-dimensional latent path and is usually a **poor inference
          strategy** for parameters. Prefer filtering (`Filter` with
          `ContinuousTime*Config`) or particle methods instead.

    Tip for speed:
        - Use `source="em_scan"` if you are happy with a simple Euler-Maruyama forward simulation
          (10–20x faster than Diffrax's implementation; see
          [Diffrax Issue #517](https://github.com/patrick-kidger/diffrax/issues/517)).
        - Use `source="diffrax"` if you want greater flexibility in the solver and step-size control.
    """

    def __init__(
        self,
        simulator_config: SDESimulatorConfig | None = None,
        *,
        solver: dfx.AbstractSolver | object = _SIMULATOR_CONFIG_UNSET,
        stepsize_controller: dfx.AbstractStepSizeController
        | object = _SIMULATOR_CONFIG_UNSET,
        adjoint: dfx.AbstractAdjoint | object = _SIMULATOR_CONFIG_UNSET,
        dt0: float | int | Array | object = _SIMULATOR_CONFIG_UNSET,
        tol_vbt: float | int | Array | None | object = _SIMULATOR_CONFIG_UNSET,
        max_steps: int | None | object = _SIMULATOR_CONFIG_UNSET,
        n_simulations: int = 1,
        source: Literal["diffrax", "em_scan"] | object = _SIMULATOR_CONFIG_UNSET,
    ):
        """Configure SDE integration settings.

        Args:
            simulator_config: Optional structured simulator config. When
                provided, use it instead of direct solver kwargs.
            solver: Diffrax solver for the SDE (e.g., [`dfx.Heun`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/)).
                For solver guidance, see [How to choose a solver](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/).
            stepsize_controller: Diffrax step-size controller. Use
                [`dfx.ConstantStepSize`](https://docs.kidger.site/diffrax/api/stepsize_controller/)
                for fixed-step simulation, or an adaptive controller for error-controlled stepping.
            adjoint: Diffrax adjoint strategy used for differentiation through the
                solver (relevant when used under gradient-based inference). See
                [Adjoints](https://docs.kidger.site/diffrax/api/adjoints/).
            dt0: Initial step size (float or JAX array) passed to
                [`diffrax.diffeqsolve`](https://docs.kidger.site/diffrax/api/diffeqsolve/).
            tol_vbt: Tolerance parameter for
                [`diffrax.VirtualBrownianTree`](https://docs.kidger.site/diffrax/api/brownian/). If None,
                defaults to `dt0 / 2`. For statistically correct simulation, this
                must be smaller than `dt0`.
            max_steps: Optional hard cap on solver steps.
            n_simulations: Number of independent trajectory simulations. When > 1,
                states and observations have an extra leading dimension (n_simulations, T, ...).
            source: SDE backend to use. `"diffrax"` uses Diffrax + Brownian tree.
                `"em_scan"` uses a custom fixed-step Euler-Maruyama `lax.scan`
                that advances at every `dt0` tick and also lands exactly on all
                requested solve times. Default is `"em_scan"` for speed.

        Notes:
            - `VirtualBrownianTree` draws randomness via `numpyro.prng_key()`, so
              `SDESimulator` must be executed inside a seeded NumPyro context.
        """
        _validate_no_config_and_direct_kwargs(
            simulator_config=simulator_config,
            config_name="simulator_config",
            direct_kwargs={
                "solver": solver,
                "stepsize_controller": stepsize_controller,
                "adjoint": adjoint,
                "dt0": dt0,
                "tol_vbt": tol_vbt,
                "max_steps": max_steps,
                "source": source,
            },
        )

        if simulator_config is None:
            solver_value: dfx.AbstractSolver = cast(
                dfx.AbstractSolver,
                dfx.Heun() if solver is _SIMULATOR_CONFIG_UNSET else solver,
            )
            stepsize_controller_value: dfx.AbstractStepSizeController = cast(
                dfx.AbstractStepSizeController,
                (
                    dfx.ConstantStepSize()
                    if stepsize_controller is _SIMULATOR_CONFIG_UNSET
                    else stepsize_controller
                ),
            )
            adjoint_value: dfx.AbstractAdjoint = cast(
                dfx.AbstractAdjoint,
                (
                    dfx.RecursiveCheckpointAdjoint()
                    if adjoint is _SIMULATOR_CONFIG_UNSET
                    else adjoint
                ),
            )
            dt0_value: float | int | Array
            if dt0 is _SIMULATOR_CONFIG_UNSET:
                dt0_value = 1e-4
            else:
                dt0_value = cast(float | int | Array, dt0)
            tol_vbt_value: float | int | Array | None
            if tol_vbt is _SIMULATOR_CONFIG_UNSET:
                tol_vbt_value = None
            else:
                tol_vbt_value = cast(float | int | Array | None, tol_vbt)
            max_steps_value: int | None
            if max_steps is _SIMULATOR_CONFIG_UNSET:
                max_steps_value = None
            else:
                max_steps_value = cast(int | None, max_steps)
            source_value: Literal["diffrax", "em_scan"] = cast(
                Literal["diffrax", "em_scan"],
                "em_scan" if source is _SIMULATOR_CONFIG_UNSET else source,
            )
            simulator_config = SDESimulatorConfig(
                solver=solver_value,
                stepsize_controller=stepsize_controller_value,
                adjoint=adjoint_value,
                dt0=dt0_value,
                tol_vbt=tol_vbt_value,
                max_steps=max_steps_value,
                source=source_value,
            )

        self.simulator_config = simulator_config
        self.diffeqsolve_settings = simulator_config.diffeqsolve_settings()
        self.n_simulations = n_simulations
        self.source = simulator_config.source
        self.tol_vbt = simulator_config.resolved_tol_vbt()

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

        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, times)
            control_path_eval: Callable[[Array], Array | None] = lambda t: (
                control_path.evaluate(t, left=False)
            )
        else:
            control_path_eval = lambda t: None

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]
        sim_keys = jr.split(rng_key, n_sim)

        def _sim_one_trajectory(key: Array, x0: Array) -> tuple[Array, Array]:
            k_solve, k_obs = jr.split(key, 2)
            states = solve_sde(
                source=self.source,
                dynamics=dynamics,
                t0=t0,
                saveat_times=times,
                x0=x0,
                control_path_eval=control_path_eval,
                diffeqsolve_settings=self.diffeqsolve_settings,
                key=k_solve,
                tol_vbt=self.tol_vbt,
            )
            observations = _emit_observations(
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
        initial_state = dynamics.initial_condition.sample(
            initial_key, sample_shape=(self.n_simulations,)
        )
        return self._simulate_forward_from_initial_state(
            dynamics,
            initial_state=jnp.asarray(initial_state),
            rng_key=rollout_key,
            times=times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )

    def _simulate(
        self,
        name: str,
        dynamics,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, Array]:
        """
        Unroll a continuous-time SDE as a NumPyro model.

        This method:
        - samples the initial latent state as `numpyro.sample("x_0", ...)`,
        - integrates the SDE to all `obs_times` using Diffrax,
        - emits observations at those times as `numpyro.sample("y_i", ..., obs=...)`,
        - and returns trajectories for deterministic recording.

        To handle controls, we use a rectilinear interpolation that is right-continuous,
        i.e., if ctrl_times = [0.0, 1.0, 2.0] and ctrl_values = [0.0, 1.0, 2.0],
        then the control at time 1.0 is the value at time 1.0.

        Args:
            dynamics: A `DynamicalModel` whose `state_evolution` is a
                `ContinuousTimeStateEvolution` with a non-None diffusion
                and inferred `bm_dim` (set during `DynamicalModel` construction).
            obs_times: Times at which to save the latent state and emit observations.
                Required.
            obs_values: Optional observation array. If provided, observation sites are
                conditioned via `obs=obs_values[i]`.
            ctrl_times: Optional control times.
            ctrl_values: Optional control values aligned to `ctrl_times`.
            predict_times: Optional prediction times. If provided, prediction sites are
                emitted at those times as `numpyro.sample("y_i", ..., obs=None)`.
        Returns:
            dict[str, State]: Dictionary with `"times"`, `"states"`, and
                `"observations"` trajectories.

        Warning:
            Conditioning on `obs_values` here is generally **not** a good way to do
            parameter inference for SDEs, because it introduces an explicit, high-
            dimensional latent path. Prefer filtering (`Filter`) or particle methods.
        """
        if not isinstance(
            dynamics.state_evolution, StochasticContinuousTimeStateEvolution
        ):
            raise NotImplementedError(
                "SDESimulator only works with StochasticContinuousTimeStateEvolution, got "
                f"{type(dynamics.state_evolution)}"
            )

        if obs_times is not None:
            raise ValueError(
                "obs_times must not be provided to an SDESimulator; it cannot be used for inference. \
                Please use a filter, or discretize the SDE and use a DiscreteTimeSimulator. \
                A natural example forthcoming (i.e., to be implemented) is the SimulatedLikelihoodDiscretizer."
            )

        if obs_values is not None:
            raise ValueError(
                "obs_values conditioning is not supported for SDESimulator. "
                "Use Filter for inference with SDEs."
            )

        times = predict_times
        if times is None:
            raise ValueError("predict_times must be provided for SDESimulator.")

        prng_key = numpyro.prng_key()
        if prng_key is None:
            raise ValueError("PRNG key required for simulation")
        with numpyro.plate(f"{name}_n_simulations", self.n_simulations):
            initial_state = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        result = self._simulate_forward_from_initial_state(
            dynamics,
            initial_state=jnp.asarray(initial_state),
            rng_key=prng_key,
            times=times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
        return _simulated_result_to_dict(result)
