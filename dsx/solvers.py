import jax
import jax.numpy as jnp
from dsx.handlers import BaseSolver, States
from dsx.dynamical_models import ContinuousTimeStateEvolution, StochasticContinuousTimeStateEvolution
from dsx.utils import dsx_to_cd_dynamax
from cd_dynamax import ContDiscreteNonlinearGaussianSSM
import diffrax as dfx
import equinox as eqx


class SDESolver(BaseSolver):
    """Solver that works with StochasticContinuousTimeStateEvolution."""
    
    def __init__(self,
                key: jax.Array,
                solver: dfx.AbstractSolver = dfx.Heun(),
                stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
                adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
                dt0: float = 0.01,
                tol_vbt: float = 1e-1, # tolerance for virtual brownian tree
                max_steps: int = 1e5,
                ):

        self.diffeqsolve_settings = {
            "key": key,
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
            "tol_vbt": tol_vbt,
            "max_steps": max_steps,
        }

    def solve(self, times, dynamics, key) -> States:
                
        if not isinstance(dynamics, StochasticContinuousTimeStateEvolution):
            raise NotImplementedError(f"SDESolver only works with StochasticContinuousTimeStateEvolution, got {type(dynamics)}")

        # Generate a CD-Dynamax-compatible parameter dict
        # Works for both stochastic and deterministic dynamics
        params = dsx_to_cd_dynamax(dynamics)
        
        # Instantiate the CD-Dynamax model (gets a solver internally, which is only used by .sample() method)
        cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(state_dim=dynamics.state_dim,
                                                            emission_dim=dynamics.observation_dim,
                                                            diffeqsolve_settings=self.diffeqsolve_settings
                                                            )

        # Sample states and emissions from the model using solver settings defined above.
        states, emissions = cd_dynamax_model.sample(
            params=params,
            key=key,
            num_timesteps=len(times),
            t_emissions=times,
            transition_type="path",
        )

        return {"states": states, "observations": emissions}

class ODESolver(BaseSolver):
    """Solver that works with ContinuousTimeStateEvolution."""
    
    def __init__(self,
                solver: dfx.AbstractSolver = dfx.Dopri5(),
                stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
                adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
                dt0: float = 0.01,
                max_steps: int = 1e5,
                ):

        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
            "max_steps": max_steps,
        }

    def solve(self, times, dynamics, key) -> States:
                
        if not isinstance(dynamics, ContinuousTimeStateEvolution):
            raise NotImplementedError(f"ODESolver only works with ContinuousTimeStateEvolution, got {type(dynamics)}")

        # Generate a CD-Dynamax-compatible parameter dict
        # Works for both stochastic and deterministic dynamics
        params = dsx_to_cd_dynamax(dynamics)
        
        # Instantiate the CD-Dynamax model (gets a solver internally, which is only used by .sample() method)
        cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(state_dim=dynamics.state_dim,
                                                            emission_dim=dynamics.observation_dim,
                                                            diffeqsolve_settings=self.diffeqsolve_settings
                                                            )

        # Sample states and emissions from the model using solver settings defined above.
        states, emissions = cd_dynamax_model.sample(
            params=params,
            key=key,
            num_timesteps=len(times),
            t_emissions=times,
            transition_type="path",
        )

        return {"states": states, "observations": emissions}
