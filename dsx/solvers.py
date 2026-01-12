from dsx.handlers import BaseSolver
from dsx.ops import States
from dsx.dynamical_models import ContinuousTimeStateEvolution
from dsx.utils import dsx_to_cd_dynamax
from cd_dynamax import ContDiscreteNonlinearGaussianSSM
import diffrax as dfx
from jax import Array
import jax.numpy as jnp
import jax.random as jr
from jax import lax


class SDESolver(BaseSolver):
    """Solver that works with ContinuousTimeStateEvolution with stochastic dynamics."""

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

    def solve(self, times, dynamics) -> States:
        if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            raise NotImplementedError(
                f"SDESolver only works with ContinuousTimeStateEvolution, got {type(dynamics.state_evolution)}"
            )

        if (
            dynamics.state_evolution.diffusion_coefficient is None
            or dynamics.state_evolution.diffusion_covariance is None
        ):
            raise ValueError(
                "SDESolver requires both diffusion_coefficient and diffusion_covariance to be "
                f"defined (got coeff={dynamics.state_evolution.diffusion_coefficient}, "
                f"cov={dynamics.state_evolution.diffusion_covariance}). "
                "Use ODESolver for deterministic dynamics."
            )

        # Generate a CD-Dynamax-compatible parameter dict
        # Works for both stochastic and deterministic dynamics
        params = dsx_to_cd_dynamax(dynamics)

        # Instantiate the CD-Dynamax model (gets a solver internally, which is only used by .sample() method)
        cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            diffeqsolve_settings=self.diffeqsolve_settings,
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
            transition_type="path",
        )

        return {"states": states, "observations": emissions}


class ODESolver(BaseSolver):
    """Solver that works with ContinuousTimeStateEvolution."""

    def __init__(
        self,
        key: Array,  # key for model randomness (initial condition and observation noise)
        solver: dfx.AbstractSolver = dfx.Dopri5(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        dt0: float = 0.01,
        max_steps: int = int(1e5),
    ):
        self.key = (
            key  # key for model randomness (initial condition and observation noise)
        )
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
            "max_steps": max_steps,
        }

    def solve(self, times, dynamics) -> States:
        if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            raise NotImplementedError(
                f"ODESolver only works with ContinuousTimeStateEvolution, got {type(dynamics.state_evolution)}"
            )

        if (
            dynamics.state_evolution.diffusion_coefficient is not None
            or dynamics.state_evolution.diffusion_covariance is not None
        ):
            raise ValueError(
                "ODESolver requires both diffusion_coefficient and diffusion_covariance to be "
                "None for deterministic dynamics. Use SDESolver for stochastic dynamics."
            )

        # Generate a CD-Dynamax-compatible parameter dict
        # Works for both stochastic and deterministic dynamics
        params = dsx_to_cd_dynamax(dynamics)

        # Instantiate the CD-Dynamax model (gets a solver internally, which is only used by .sample() method)
        cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(
            state_dim=dynamics.state_dim,
            emission_dim=dynamics.observation_dim,
            diffeqsolve_settings=self.diffeqsolve_settings,
        )

        # ensure that times has shape (num_timesteps, 1)
        if times.ndim == 1:
            times = times[:, None]

        # Sample states and emissions from the model using solver settings defined above.
        states, emissions = cd_dynamax_model.sample(
            params=params,
            key=self.key,
            num_timesteps=len(times),
            t_emissions=times,
            transition_type="path",
        )

        return {"states": states, "observations": emissions}


class DiscreteTimeSolver(BaseSolver):
    """Solver for discrete-time state evolution models using lax.scan."""

    # TODO: add controls

    def __init__(self, key: Array):
        self.key = key

    def solve(self, times, dynamics) -> States:
        key = self.key

        # --- Sample initial state ---
        key, subkey = jr.split(key)
        x0 = dynamics.initial_condition.sample(subkey)
        key, subkey = jr.split(key)
        y0 = (
            dynamics.observation_model(x=x0, u=None, t=times[0]).sample(subkey)
            if dynamics.observation_model is not None
            else None
        )

        def step(carry, t):
            key, x_t = carry

            # --- Transition ---
            key, subkey = jr.split(key)
            transition_dist = dynamics.state_evolution(x=x_t, u=None, t=t)
            x_next = transition_dist.sample(subkey)

            # --- Observation (optional) ---
            if dynamics.observation_model is not None:
                key, subkey = jr.split(key)
                obs_dist = dynamics.observation_model(x=x_next, u=None, t=t)
                y_next = obs_dist.sample(subkey)
            else:
                y_next = None

            return (key, x_next), (x_next, y_next)

        # Run scan over times[:-1] (since x0 is already sampled)
        (_, _), (xs, ys) = lax.scan(
            step,
            (key, x0),
            times[:-1],
        )

        # Prepend initial state
        states = jnp.concatenate([x0[None, ...], xs], axis=0)

        if dynamics.observation_model is not None:
            observations = jnp.concatenate([y0[None, ...], ys], axis=0)
            return {
                "states": states,
                "observations": observations,
            }
        else:
            return {
                "states": states,
            }
