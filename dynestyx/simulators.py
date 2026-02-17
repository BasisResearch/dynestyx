import dataclasses
from collections.abc import Callable

import diffrax as dfx
import jax.numpy as jnp
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from jax import Array
from numpyro.contrib.control_flow import scan as nscan

from dynestyx.dynamical_models import (
    Context,
    ContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    State,
)
from dynestyx.handlers import BaseSimulator
from dynestyx.observations import DiracIdentityObservation
from dynestyx.utils import (
    _get_controls,
    _get_val_or_None,
    _validate_control_dim,
)

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


class SDESimulator(BaseSimulator):
    """Simulator that works with ContinuousTimeStateEvolution with stochastic dynamics."""

    def __init__(
        self,
        key: Array,
        solver: dfx.AbstractSolver = dfx.Heun(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        dt0: float = 1e-4,
        tol_vbt: float | None = None,
        max_steps: int | None = None,
    ):
        self.key = key  # key for model randomness (initial condition, SDE solver, and observation noise)
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
            "max_steps": max_steps,
        }

        if tol_vbt is None:
            self.tol_vbt = dt0 / 2.0
        else:
            self.tol_vbt = tol_vbt

    def simulate(self, context: Context, dynamics) -> dict[str, State]:
        if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            raise NotImplementedError(
                f"SDESimulator only works with ContinuousTimeStateEvolution, got {type(dynamics.state_evolution)}"
            )

        if (
            dynamics.state_evolution.diffusion_coefficient is None
            or dynamics.state_evolution.bm_dim is None
        ):
            raise ValueError(
                "SDESimulator requires both diffusion_coefficient and bm_dim to be "
                f"defined (got coeff={dynamics.state_evolution.diffusion_coefficient}, "
                f"bm_dim={dynamics.state_evolution.bm_dim}). "
                "Use ODESimulator for deterministic dynamics."
            )

        # Extract times from context
        if context.observations is None or context.observations.times is None:
            raise ValueError("context.observations.times must be provided")

        obs_traj = context.observations
        times = obs_traj.times
        obs_values = obs_traj.values

        if times is None:
            raise ValueError("times must be provided")

        # Extract controls from context if available
        ctrl_times, ctrl_values = _get_controls(context, times)

        # Validate that control_dim is set when controls are present
        _validate_control_dim(dynamics, ctrl_values)

        with numpyro.handlers.seed(rng_seed=self.key):
            initial_state = numpyro.sample("x_0", dynamics.initial_condition)

            if ctrl_times is not None and ctrl_values is not None:
                # We use rectilinear interpolation, to match cd_dynamax
                _ct, _cv = dfx.rectilinear_interpolation(ts=ctrl_times, ys=ctrl_values)
                control_path = dfx.LinearInterpolation(ts=_ct, ys=_cv)
                control_path_eval: Callable[[Array], Array | None] = (
                    control_path.evaluate
                )
            else:
                control_path_eval = lambda t: None

            def _drift(t, y, args):
                u_t = control_path_eval(t)
                return dynamics.state_evolution.drift(x=y, u=u_t, t=t)

            def _diffusion(t, y, args):
                u_t = control_path_eval(t)
                return dynamics.state_evolution.diffusion_coefficient(x=y, u=u_t, t=t)

            bm = dfx.VirtualBrownianTree(
                t0=times[0],
                t1=times[-1],
                tol=self.tol_vbt,
                shape=(dynamics.state_evolution.bm_dim,),
                key=numpyro.prng_key(),
            )

            terms = dfx.MultiTerm(  # type: ignore
                dfx.ODETerm(_drift), dfx.ControlTerm(_diffusion, bm)
            )

            sol = dfx.diffeqsolve(
                terms,
                t0=times[0],
                t1=times[-1],
                y0=initial_state,
                args=None,
                saveat=dfx.SaveAt(ts=times),
                **self.diffeqsolve_settings,
            )
            states_sol = sol.ys  # (T, ..., state_dim)

            def _create_observations_step(carry, t_idx):
                x_t = states_sol[t_idx]
                t = times[t_idx]
                u_t = _get_val_or_None(ctrl_values, t_idx)
                y_t = numpyro.sample(
                    f"y_{t_idx}",
                    dynamics.observation_model(x=x_t, u=u_t, t=t),
                    obs=_get_val_or_None(obs_values, t_idx),
                )
                return carry, y_t

        states = states_sol
        _, emissions = nscan(_create_observations_step, None, jnp.arange(len(times)))

        return {"times": times, "states": states, "observations": emissions}


@dataclasses.dataclass
class DiscreteTimeSimulator(BaseSimulator):
    """Simulator for discrete-time dynamical models.

    Assumes we have ic, transition, and observation distributions,
    as well as (time_index, observation) pairs in the context.

    Simply unrolls the model and adds obs=data as you would in numpyro.

    This does not explicitly add logfactors; it lets numpyro do it automatically.
    Instead, it just unrolls the model and adds observed sites
    (which numpyro uses to compute logfactors).
    """

    def simulate(
        self,
        context: Context,
        dynamics: DynamicalModel,
    ) -> dict[str, State]:
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")
        obs_values = obs_traj.values

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs_times)

        T = len(obs_times)
        if T < 1:
            raise ValueError("obs_times must contain at least one timepoint")

        # DiracIdentityObservation with observed values: y_t = x_t, so we use plating
        # instead of scan. state_evolution returns a dist; call it with batched inputs.
        if isinstance(dynamics.observation_model, DiracIdentityObservation) and (
            obs_values is not None
        ):
            numpyro.sample("x_0", dynamics.initial_condition, obs=obs_values[0])
            numpyro.deterministic("y_0", obs_values[0])
            if T == 1:
                # No transitions exist for a single-timepoint trajectory.
                return {
                    "times": obs_times,
                    "states": obs_values,
                    "observations": obs_values,
                }

            # Ensure (T-1, state_dim) so swapaxes to (state_dim, T-1) is valid (state_dim=1 => 1D otherwise).
            if obs_values.ndim == 1:
                x_prev = obs_values[:-1][:, None]
                x_next = obs_values[1:][:, None]
            else:
                x_prev = obs_values[:-1]
                x_next = obs_values[1:]
            if ctrl_values is not None:
                if ctrl_values.ndim == 1:
                    u_prev = ctrl_values[:-1][:, None]
                else:
                    u_prev = ctrl_values[:-1]
            else:
                u_prev = None
            t_now = obs_times[:-1]
            t_next = obs_times[1:]

            # Pass state (and controls) with batch as last axis so drift can use
            # naive indexing (x[0], x[1], ...) and discretizer broadcasts correctly.
            x_prev_batch_last = jnp.swapaxes(x_prev, 0, 1)
            x_next_batch_last = jnp.swapaxes(x_next, 0, 1)
            u_prev_batch_last = (
                jnp.swapaxes(u_prev, 0, 1) if u_prev is not None else None
            )

            with numpyro.plate("time", T - 1):
                trans = dynamics.state_evolution(
                    x_prev_batch_last,
                    u_prev_batch_last,
                    t_now,
                    t_next,  # type: ignore
                )
                # obs shape must match trans.batch_shape + trans.event_shape: use
                # time-first (T-1, state_dim) for e.g. discretizer; batch-last (state_dim, T-1) for scalar.
                obs_next = x_next_batch_last if dynamics.state_dim == 1 else x_next
                numpyro.sample("x_next", trans, obs=obs_next)  # type: ignore

            return {
                "times": obs_times,
                "states": obs_values,
                "observations": obs_values,
            }

        # Default: scan over time
        # Sample initial state
        x_prev: State = numpyro.sample("x_0", dynamics.initial_condition)  # type: ignore

        # sample initial observation
        u_0 = _get_val_or_None(ctrl_values, 0)
        y_0 = numpyro.sample(
            "y_0",
            dynamics.observation_model(x_prev, u_0, obs_times[0]),
            obs=_get_val_or_None(obs_values, 0),
        )

        def _step(x_prev, t_idx):
            t_now = obs_times[t_idx]
            t_next = obs_times[t_idx + 1]
            u_now = _get_val_or_None(ctrl_values, t_idx)
            u_next = _get_val_or_None(ctrl_values, t_idx + 1)
            # Sample next state
            x_t = numpyro.sample(
                f"x_{t_idx + 1}",
                dynamics.state_evolution(x=x_prev, u=u_now, t_now=t_now, t_next=t_next),
            )

            # Sample observation
            y_t = numpyro.sample(
                f"y_{t_idx + 1}",
                dynamics.observation_model(x=x_t, u=u_next, t=t_next),
                obs=_get_val_or_None(obs_values, t_idx + 1),
            )
            return x_t, (x_t, y_t)

        # Run scan and collect states and observations
        # scan_outputs will be (scan_states, scan_observations) where each is shape (T-1, ...)
        _, scan_outputs = nscan(_step, x_prev, jnp.arange(T - 1))
        scan_states, scan_observations = scan_outputs

        # Stack initial state/observation with scanned results
        # x_prev is shape (state_dim,) or scalar, scan_states is (T-1, state_dim)
        # y_0 is shape (obs_dim,) or scalar, scan_observations is (T-1, obs_dim)
        # Use expand_dims to ensure proper shape for concatenation
        # shape (1, state_dim) or (1,)
        x_0_expanded = jnp.expand_dims(x_prev, axis=0)  # type: ignore
        y_0_expanded = jnp.expand_dims(y_0, axis=0)  # shape (1, obs_dim) or (1,)
        states = jnp.concatenate(
            [x_0_expanded, scan_states], axis=0
        )  # shape (T, state_dim)
        observations = jnp.concatenate(
            [y_0_expanded, scan_observations], axis=0
        )  # shape (T, obs_dim)

        return {"times": obs_times, "states": states, "observations": observations}


@dataclasses.dataclass
class ODESimulator(BaseSimulator):
    """Simulator for continuous-time deterministic (ODE) dynamical models.

    Assumes we have ic, transition, and observation distributions,
    as well as (time_index, observation) pairs in the context.

    Simply unrolls the model and adds obs=data as you would in numpyro.

    This does not add logfactors, just unrolls the model and adds observed sites.
    """

    solver: dfx.AbstractSolver = dfx.Tsit5()
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize()
    dt0: float = 0.01
    max_steps: int = 10_000

    def __init__(
        self,
        solver: dfx.AbstractSolver = dfx.Tsit5(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        dt0: float = 0.01,
        max_steps: int = 10_000,
    ):
        self.solver = solver
        self.adjoint = adjoint
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps

    def simulate(
        self,
        context: Context,
        dynamics: DynamicalModel,
    ) -> dict[str, State]:
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        obs_values = obs_traj.values
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs_times)

        T = len(obs_times)

        # Sample initial state
        x_prev = numpyro.sample("x_0", dynamics.initial_condition)

        # Create drift function that interpolates controls
        if ctrl_times is not None and ctrl_values is not None:
            # Create LinearInterpolation for controls using diffrax
            # We use rectilinear interpolation, to match cd_dynamax
            _ct, _cv = dfx.rectilinear_interpolation(ts=ctrl_times, ys=ctrl_values)
            control_path = dfx.LinearInterpolation(ts=_ct, ys=_cv)

            def f(t, y, args):
                # Evaluate control at time t using interpolation
                u_t = control_path.evaluate(t)
                return dynamics.state_evolution.drift(x=y, u=u_t, t=t)

        else:

            def f(t, y, args):
                return dynamics.state_evolution.drift(x=y, u=None, t=t)

        # Solve ODE at all observation times using diffrax
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(f),
            solver=self.solver,
            t0=obs_times[0],
            t1=obs_times[-1],
            dt0=self.dt0,
            y0=x_prev,
            saveat=dfx.SaveAt(ts=obs_times),
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )
        x_sol = sol.ys  # shape (T, state_dim) # includes initial state at t0

        # use scan to sample observations and collect them
        def _step(carry, t_idx):
            x_t = x_sol[t_idx]
            t = obs_times[t_idx]
            u_t = _get_val_or_None(ctrl_values, t_idx)
            # Sample observation
            y_t = numpyro.sample(
                f"y_{t_idx}",
                dynamics.observation_model(x=x_t, u=u_t, t=t),
                obs=_get_val_or_None(obs_values, t_idx),
            )
            return carry, y_t

        # Run scan and collect observations
        # scan_outputs will be observations of shape (T, obs_dim)
        _, scan_observations = nscan(_step, None, jnp.arange(T))
        observations = scan_observations  # shape (T, obs_dim)

        return {"times": obs_times, "states": x_sol, "observations": observations}


class Simulator(BaseSimulator):
    """Simulator for dynamical models.

    This is a wrapper class that selects the appropriate simulator based on the type of dynamical model.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.simulator = None

    def simulate(self, context: Context, dynamics: DynamicalModel) -> dict[str, State]:
        if self.simulator is None:
            raise ValueError("Simulator not initialized. This shouldn't happen.")

        return self.simulator.simulate(context, dynamics)

    def add_solved_sites(self, name: str, dynamics: DynamicalModel, context: Context):
        if self.simulator is None:
            if isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
                if (
                    dynamics.state_evolution.diffusion_coefficient is None
                    or dynamics.state_evolution.bm_dim is None
                ):
                    self.simulator = ODESimulator(*self.args, **self.kwargs)
                else:
                    self.simulator = SDESimulator(*self.args, **self.kwargs)
            elif isinstance(dynamics.state_evolution, DiscreteTimeStateEvolution):
                self.simulator = DiscreteTimeSimulator(*self.args, **self.kwargs)
            else:
                raise ValueError(
                    f"Unsupported state evolution type: {type(dynamics.state_evolution)}."
                    + "If using a generic function as a state evolution, you must specify the type of simulator manually."
                )

        return self.simulator.add_solved_sites(name, dynamics, context)
