from typing import Any

import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM

from dynestyx.dynamical_models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    LinearGaussianStateEvolution,
)
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

        def emission_function(x, u, t):
            if x.ndim > 1:
                return x @ obs.H.T + (
                    obs.D @ u if obs.D is not None and u is not None else 0
                )
            else:
                return obs.H @ x + (
                    obs.D @ u if obs.D is not None and u is not None else 0
                )

        params.update(
            {
                "emission_function": emission_function,
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


def lti_to_nlgssm_params(dynamics: DynamicalModel) -> ParamsNLGSSM:
    """Build ParamsNLGSSM from an LTI DynamicalModel (LinearGaussianStateEvolution + LinearGaussianObservation).

    Used by EKF/UKF in discrete_time_filters. Dynamics f(x,u) = A@x + b + B@u, emissions h(x,u) = H@x + d + D@u.
    When control_dim is 0, u has shape (0,) and the B@u / D@u terms are omitted.
    """
    if not isinstance(
        dynamics.state_evolution, LinearGaussianStateEvolution
    ) or not isinstance(dynamics.observation_model, LinearGaussianObservation):
        raise TypeError(
            "lti_to_nlgssm_params expects DynamicalModel with "
            "LinearGaussianStateEvolution and LinearGaussianObservation."
        )
    if not isinstance(dynamics.initial_condition, dist.MultivariateNormal):
        raise TypeError(
            "lti_to_nlgssm_params expects initial_condition to be MultivariateNormal."
        )
    evo = dynamics.state_evolution
    obs = dynamics.observation_model
    ic = dynamics.initial_condition
    state_dim = dynamics.state_dim
    control_dim = dynamics.control_dim

    F = evo.A
    Q = evo.cov
    b = evo.bias if evo.bias is not None else jnp.zeros(state_dim)
    B = evo.B if evo.B is not None else jnp.zeros((state_dim, control_dim))

    H = obs.H
    R = obs.R
    d = obs.bias if obs.bias is not None else jnp.zeros(dynamics.observation_dim)
    D = (
        obs.D
        if obs.D is not None
        else jnp.zeros((dynamics.observation_dim, control_dim))
    )

    def dynamics_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        out = F @ x + b
        if control_dim > 0 and u.size > 0:
            out = out + (B @ jnp.reshape(u, (-1, 1))).ravel()
        return out

    def emission_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        out = H @ x + d
        if control_dim > 0 and u.size > 0:
            out = out + (D @ jnp.reshape(u, (-1, 1))).ravel()
        return out

    return ParamsNLGSSM(
        initial_mean=ic.loc,  # type: ignore[attr-defined]
        initial_covariance=ic.covariance_matrix,  # type: ignore[attr-defined]
        dynamics_function=dynamics_function,
        dynamics_covariance=Q,
        emission_function=emission_function,
        emission_covariance=R,
    )
