from typing import Any

import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM

from dynestyx.dynamical_models import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.observations import LinearGaussianObservation

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


def dsx_to_cd_dynamax(
    dsx_model: DynamicalModel, cd_model: SSMType | None = None
) -> tuple[dict, bool]:
    """
    Maps a dsx Dynamical Model to a CD-Dynamax-compatible model.
    """

    params: dict[str, Any] = {}

    ## Map state evolution ##
    state_evo = dsx_model.state_evolution
    if isinstance(state_evo, ContinuousTimeStateEvolution):
        if state_evo.drift is not None:
            params.update(
                {
                    "drift": state_evo.drift,
                }
            )
        else:
            raise ValueError(
                "drift is None; default drift (e.g., ZERO) is not yet handled carefully."
            )
        if state_evo.diffusion_coefficient is not None:
            params.update(
                {
                    "diffusion_coeff": state_evo.diffusion_coefficient,
                }
            )
        if state_evo.diffusion_covariance is not None:
            params.update(
                {
                    "diffusion_cov": state_evo.diffusion_covariance,
                }
            )
    else:
        raise NotImplementedError(
            f"State evolution of type {type(state_evo)} is not supported yet."
        )

    ## Map initial condition ##
    ic = dsx_model.initial_condition
    if isinstance(ic, dist.MultivariateNormal):
        params.update(
            {
                "initial_mean": ic.loc,  # type: ignore
                "initial_cov": ic.covariance_matrix,
            }
        )
    elif isinstance(ic, dist.Normal):
        params.update({"initial_mean": ic.loc, "initial_cov": jnp.square(ic.scale)})  # type: ignore
    else:
        raise NotImplementedError(
            f"Initial condition of type {type(ic)} is not supported yet."
        )

    ## Map observation model ##
    obs = dsx_model.observation_model
    non_gaussian_flag = False
    if isinstance(obs, LinearGaussianObservation):
        params.update(
            {
                "emission_function": lambda x, u, t: x @ obs.H.T
                if x.ndim > 1
                else obs.H @ x,
                "emission_cov": obs.R,  # type: ignore
            }
        )
    else:
        # TODO: check for linear-gaussian observation models and extract H, R
        # TODO: check for Gaussian observation and use CDNLGSSM
        non_gaussian_flag = True
        params.update(emission_distribution=dsx_model.observation_model)
        # raise NotImplementedError(
        #     f"Observation model of type {type(obs)} is not supported yet."
        # )

    if cd_model is None:
        if non_gaussian_flag:
            model_to_use: SSMType = ContDiscreteNonlinearSSM(
                state_dim=dsx_model.state_dim,
                emission_dim=dsx_model.observation_dim,
                input_dim=dsx_model.control_dim,
            )
        else:
            model_to_use = ContDiscreteNonlinearGaussianSSM(
                state_dim=dsx_model.state_dim,
                emission_dim=dsx_model.observation_dim,
                input_dim=dsx_model.control_dim,
            )
    else:
        model_to_use = cd_model

    cd_dynamax_params = model_to_use.build_params(**params)

    return cd_dynamax_params, non_gaussian_flag
