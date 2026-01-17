from dsx.handlers import BaseSolver
from dsx.ops import States, Context
from dsx.dynamical_models import ContinuousTimeStateEvolution
from dsx.utils import dsx_to_cd_dynamax
from cd_dynamax import ContDiscreteNonlinearGaussianSSM
import diffrax as dfx
from jax import Array


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

    def solve(self, context: Context, dynamics) -> States:
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

        # Extract times from context
        if context.observations is None or context.observations.times is None:
            raise ValueError("context.observations.times must be provided")
        times = context.observations.times

        # Extract controls from context if available
        ctrl_traj = context.controls
        ctrl_times = ctrl_traj.times if ctrl_traj is not None else None
        ctrl_values = ctrl_traj.values if ctrl_times is not None else None

        # Validate controls are Array (not dict) if provided
        if isinstance(ctrl_values, dict):
            raise ValueError("ctrl_values must be an Array or None, not a dict")

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
            inputs=ctrl_values,
            transition_type="path",
        )

        return {"times": times, "states": states, "observations": emissions}
