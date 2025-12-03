import jax
import jax.numpy as jnp

from dsx.dynamical_models import DynamicalModel, ContinuousTimeStateEvolution, StochasticContinuousTimeStateEvolution
from dsx.observations import LinearGaussianObservation
from numpyro import distributions as dist

from cd_dynamax import ContDiscreteNonlinearGaussianSSM as CDNLGSSM

def dsx_to_cd_dynamax(dsx_model: DynamicalModel) -> dict:
    """
    Maps a dsx Dynamical Model to a CD-Dynamax-compatible model.
    """
    
    params = {}
    
    ## Map state evolution ##
    state_evo = dsx_model.state_evolution
    if isinstance(state_evo, ContinuousTimeStateEvolution):
        params.update({
            'drift': state_evo.drift,
        })    
        if isinstance(state_evo, StochasticContinuousTimeStateEvolution):
            params.update({
                'diffusion_coeff': state_evo.diffusion_coefficient,
                'diffusion_cov': state_evo.diffusion_covariance,
            })
    else:
        raise NotImplementedError(f"State evolution of type {type(state_evo)} is not supported yet.")
    
    ## Map initial condition ##
    ic = dsx_model.initial_condition
    if isinstance(ic, dist.MultivariateNormal):
        params.update({
            'initial_mean': ic.loc,
            'initial_cov': ic.covariance_matrix,
        })
    elif isinstance(ic, dist.Normal):
        params.update({
            'initial_mean': ic.loc,
            'initial_cov': jnp.square(ic.scale)
        })
    else:
        raise NotImplementedError(f"Initial condition of type {type(ic)} is not supported yet.")

    ## Map observation model ##
    obs = dsx_model.observation_model
    if isinstance(obs, LinearGaussianObservation):
        params.update({
            'emission_function': lambda x, u, t: obs.H @ x,
            'emission_cov': obs.R,
        })
    else:
        raise NotImplementedError(f"Observation model of type {type(obs)} is not supported yet.")

    
    cdnlgssm = CDNLGSSM(state_dim=dsx_model.state_dim,
                            emission_dim=dsx_model.observation_dim)
    cd_dynamax_params = cdnlgssm.build_params(**params)

    return cd_dynamax_params
