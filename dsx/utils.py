import jax
import jax.numpy as jnp

from dsx.dynamical_models import DynamicalModel, ContinuousTimeStateEvolution, StochasticContinuousTimeStateEvolution
from dsx.observations import LinearGaussianObservation
from numpyro import distributions as dist

def dsx_to_cd_dynamax(dsx_model: DynamicalModel) -> dict:
    """
    Maps a dsx Dynamical Model to a CD-Dynamax-compatible model.
    """
    
    params = {}
    
    ## Map state evolution ##
    if isinstance(dsx_model.state_evolution, ContinuousTimeStateEvolution):
        params.update({
            'dynamics_drift': dsx_model.drift,
        })    
        if isinstance(dsx_model.state_evolution, StochasticContinuousTimeStateEvolution):
            params.update({
                'dynamics_diffusion_coefficient': dsx_model.diffusion_coefficient,
                'dynamics_diffusion_cov': dsx_model.diffusion_covariance,
            })
    else:
        raise NotImplementedError(f"State evolution of type {type(dsx_model.state_evolution)} is not supported yet.")
    
    ## Map initial condition ##
    if isinstance(dsx_model.initial_condition, dist.MultivariateNormal):
        params.update({
            'initial_mean': dsx_model.initial_condition.loc,
            'initial_cov': dsx_model.initial_condition.covariance_matrix,
        })
    elif isinstance(dsx_model.initial_condition, dist.Normal):
        params.update({
            'initial_mean': dsx_model.initial_condition.loc,
            'initial_cov': jnp.square(dsx_model.initial_condition.scale)
        })
    else:
        raise NotImplementedError(f"Initial condition of type {type(dsx_model.initial_condition)} is not supported yet.")

    ## Map observation model ##
    if isinstance(dsx_model.observation_model, LinearGaussianObservation):
        params.update({
            'emission_function': dsx_model.observation_model.loc,
            'observation_cov': dsx_model.observation_model.covariance_matrix,
        })
    else:
        raise NotImplementedError(f"Observation model of type {type(dsx_model.observation_model)} is not supported yet.")

    return params