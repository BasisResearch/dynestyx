import jax
import jax.numpy as jnp
import jax.random as jr
from dsx.ops import Trajectory
from dsx.handlers import BaseCDDynamaxLogFactorAdder
from dsx.dynamical_models import ContinuousTimeStateEvolution, StochasticContinuousTimeStateEvolution
from dsx.utils import dsx_to_cd_dynamax
from cd_dynamax import ContDiscreteNonlinearGaussianSSM
import numpyro
# import diffrax as dfx
# from effectful.ops.semantics import handler, fwd, coproduct
from typing import Optional
import dataclasses

@dataclasses.dataclass
class FilterBasedMarginalLogLikelihood(BaseCDDynamaxLogFactorAdder):
    """ Log factor adder that computes marginal log likelihood via CD-Dynamax filtering."""

    key = None
    filter_type: str = "EnKF"
    filter_state_order: str = "first"
    filter_emission_order: str = "first"
    filter_num_iter: int = 1
    filter_state_cov_rescaling: float = 1.0
    filter_dt_average: float = 0.1
    enkf_N_particles: int = 25
    enkf_inflation_delta: float = 0.0
    diffeqsolve_max_steps: int = 1000
    diffeqsolve_dt0: float = 0.01
    output_fields = None
    diffeqsolve_kwargs: dict = dataclasses.field(default_factory=dict)
    extra_filter_kwargs: dict = dataclasses.field(default_factory=dict)
    warn: bool = True


    def add_log_factors(self, dynamics: StochasticContinuousTimeStateEvolution, obs: Trajectory, name: Optional[str] = "filter_marginal_log_likelihood"):
        # Do I need any of this fwd stuff here?
        # Get trajectory function from solver via fwd()
        # trajectory_fn = fwd()
        
        # Extract observed times and states
        obs_times, obs_states = obs
        
        # Generate a CD-Dynamax-compatible parameter dict
        params = dsx_to_cd_dynamax(dynamics)
        # Instantiate the CD-Dynamax model
        cd_dynamax_model = ContDiscreteNonlinearGaussianSSM(state_dim=dynamics.state_dim,
                                                            emission_dim=dynamics.observation_dim,
                                                            )
        # Compute the marginal log likelihood via filtering
        filtered = cd_dynamax_model.filter(
            params=params,
            emissions=obs_states,
            t_emissions=obs_times,
            key=numpyro.prng_key() if self.key is None else self.key,
            filter_type=self.filter_type,
            filter_state_order=self.filter_state_order,
            filter_emission_order=self.filter_emission_order,
            filter_num_iter=self.filter_num_iter,
            filter_state_cov_rescaling=self.filter_state_cov_rescaling,
            filter_dt_average=self.filter_dt_average,
            enkf_N_particles=self.enkf_N_particles,
            enkf_inflation_delta=self.enkf_inflation_delta,
            diffeqsolve_max_steps=self.diffeqsolve_max_steps,
            diffeqsolve_dt0=self.diffeqsolve_dt0,
            diffeqsolve_kwargs=self.diffeqsolve_kwargs,
            extra_filter_kwargs=self.extra_filter_kwargs,
            output_fields=self.output_fields,
            warn=self.warn
        )

        # Add the marginal log likelihood as a numpyro factor
        numpyro.factor(name, filtered.marginal_log_likelihood)
