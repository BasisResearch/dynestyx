import jax.numpy as jnp
import dataclasses

from dsx.handlers import BaseSimulator
from dsx.ops import States, Context
from dsx.dynamical_models import ContinuousTimeStateEvolution, DynamicalModel, State
from dsx.observations import DiracIdentityObservation
from dsx.utils import (
    dsx_to_cd_dynamax,
    _get_controls,
    _validate_control_dim,
    _get_val_or_None,
)
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
import diffrax as dfx
from jax import Array
import numpyro
from numpyro.contrib.control_flow import scan as nscan
import warnings

from typing import TypeAlias

SSMType: TypeAlias = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


class SDESimulator(BaseSimulator):
    """Simulator that works with ContinuousTimeStateEvolution with stochastic dynamics."""

    def __init__(
        self,
        key: Array,
        solver: dfx.AbstractSolver = dfx.Heun(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        dt0: float = 0.01,
        tol_vbt: float = 1e-1,  # tolerance for virtual brownian tree
        max_steps: int = int(1e5),
    ):
        self.key = key  # key for model randomness (initial condition, SDE solver, and observation noise)
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
            "tol_vbt": tol_vbt,
            "max_steps": max_steps,
        }

    def simulate(self, context: Context, dynamics) -> States:
        if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            raise NotImplementedError(
                f"SDESimulator only works with ContinuousTimeStateEvolution, got {type(dynamics.state_evolution)}"
            )

        if (
            dynamics.state_evolution.diffusion_coefficient is None
            or dynamics.state_evolution.diffusion_covariance is None
        ):
            raise ValueError(
                "SDESimulator requires both diffusion_coefficient and diffusion_covariance to be "
                f"defined (got coeff={dynamics.state_evolution.diffusion_coefficient}, "
                f"cov={dynamics.state_evolution.diffusion_covariance}). "
                "Use ODESimulator for deterministic dynamics."
            )

        # Extract times from context
        if context.observations is None or context.observations.times is None:
            raise ValueError("context.observations.times must be provided")
        times = context.observations.times

        # Extract controls from context if available
        ctrl_times, ctrl_values = _get_controls(context, times)

        # Validate that control_dim is set when controls are present
        _validate_control_dim(dynamics, ctrl_values)

        # Generate a CD-Dynamax-compatible parameter dict
        # Works for both stochastic and deterministic dynamics
        params, non_gaussian_flag = dsx_to_cd_dynamax(dynamics)

        # Instantiate the CD-Dynamax model (gets a solver internally, which is only used by .sample() method)
        if non_gaussian_flag:
            cd_dynamax_model: SSMType = ContDiscreteNonlinearSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
                input_dim=dynamics.control_dim,
                diffeqsolve_settings=self.diffeqsolve_settings,
            )
        else:
            cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(
                state_dim=dynamics.state_dim,
                emission_dim=dynamics.observation_dim,
                input_dim=dynamics.control_dim,
            )

        # Sample states and emissions from the model using solver settings defined above.
        # ensure that times has shape (num_timesteps, 1)
        if times.ndim == 1:
            times = times[:, None]
        states, emissions = cd_dynamax_model.sample(
            params=params,
            key=self.key,
            num_timesteps=len(times),
            t_emissions=times,
            inputs=ctrl_values,
            transition_type="path",
        )

        return {"times": times, "states": states, "observations": emissions}

    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Extract observed trajectory (original 1D times)
        obs_traj = context.observations
        if obs_traj is None or obs_traj.times is None:
            raise ValueError("context.observations.times must be provided")
        obs_times = obs_traj.times
        obs_values = obs_traj.values if obs_traj is not None else None
        if isinstance(obs_values, dict):
            raise ValueError("obs_values must be an Array or None, not a dict")

        # Controls aligned with observed times
        _, ctrl_values = _get_controls(context, obs_times)

        # Run the simulator to obtain states/emissions (times reshaped as needed for cd-dynamax)
        simulated = self.simulate(context, dynamics)
        states = simulated["states"]
        emissions = simulated["observations"]

        # Deterministic sites for times, states, emissions
        numpyro.deterministic("times", obs_times)
        numpyro.deterministic("states", states)
        numpyro.deterministic("observations", emissions)

        # Observation sample sites using the model's observation distribution only when obs values are provided.
        # If obs_values is None and emissions were produced, we avoid resampling to keep consistency with cd-dynamax.
        if obs_values is not None:
            T = len(obs_times)
            warnings.warn(
                "Adding observation sample sites in the SDESimulator will not result in proper state inference. While it provides a technically unbiased estimate of the marginal likelihood for system identification, it is highly inefficient and is not recommended."
            )
            for t_idx in range(T):
                u_t = _get_val_or_None(ctrl_values, t_idx)
                t = obs_times[t_idx]
                numpyro.sample(
                    f"y_{t_idx}",
                    dynamics.observation_model(states[t_idx], u_t, t),
                    obs=_get_val_or_None(obs_values, t_idx),
                )


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
    ) -> States:
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")
        if isinstance(obs_traj.values, dict):
            raise ValueError("obs_traj.values must be an Array or None, not a dict")
        obs_values = obs_traj.values

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs_times)

        T = len(obs_times)

        # DiracIdentityObservation with observed values: y_t = x_t, so we use plating
        # instead of scan. state_evolution returns a dist; call it with batched inputs.
        if isinstance(dynamics.observation_model, DiracIdentityObservation) and (
            obs_values is not None
        ):
            numpyro.sample("x_0", dynamics.initial_condition, obs=obs_values[0])
            numpyro.deterministic("y_0", obs_values[0])

            x_prev = obs_values[:-1]
            x_next = obs_values[1:]
            u_prev = ctrl_values[:-1] if ctrl_values is not None else None
            t_now = obs_times[:-1]
            t_next = obs_times[1:]

            with numpyro.plate("time", T - 1):
                trans = dynamics.state_evolution(
                    x_prev,
                    u_prev,
                    t_now,
                    t_next,  # type: ignore
                )
                numpyro.sample("x_next", trans, obs=x_next)  # type: ignore

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

    def simulate(
        self,
        context: Context,
        dynamics: DynamicalModel,
    ) -> States:
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        obs_values = obs_traj.values
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")
        if isinstance(obs_values, dict):
            raise ValueError("obs_values must be an Array or None, not a dict")

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs_times)

        T = len(obs_times)

        # Sample initial state
        x_prev = numpyro.sample("x_0", dynamics.initial_condition)

        # Create drift function that interpolates controls
        # For now, use piecewise constant (nearest neighbor) interpolation
        if ctrl_times is not None and ctrl_values is not None:
            # Create LinearInterpolation for controls using diffrax
            control_path = dfx.LinearInterpolation(ts=ctrl_times, ys=ctrl_values)

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
