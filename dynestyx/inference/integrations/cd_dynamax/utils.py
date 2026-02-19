from typing import Any

import jax.numpy as jnp
import numpyro.distributions as dist

from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.models import ParamsNLGSSM
from dynestyx.dynamical_models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    GaussianStateEvolution,
    LinearGaussianStateEvolution,
)
from dynestyx.observations import GaussianObservation, LinearGaussianObservation

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
        if state_evo.bm_dim is not None:
            params.update(
                {
                    "diffusion_cov": jnp.eye(state_evo.bm_dim),
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


def gaussian_to_nlgssm_params(dynamics: DynamicalModel) -> ParamsNLGSSM:
    """Build ParamsNLGSSM from a Gaussian discrete-time DynamicalModel.

    Supports linear or nonlinear dynamics:
    - `LinearGaussianStateEvolution` (LTI: f(x,u) = A@x + b + B@u with Gaussian noise),
    - `GaussianStateEvolution` (nonlinear: F(x, u, t_now, t_next) with Gaussian noise),
    with either `LinearGaussianObservation` (H x + d + D u) or `GaussianObservation`
    (arbitrary h(x,u,t) with Gaussian noise).

    Used by EKF/UKF in discrete filters. When control_dim is 0, u has shape (0,)
    and the B@u / D@u terms are omitted.
    """
    if not isinstance(
        dynamics.state_evolution, (LinearGaussianStateEvolution, GaussianStateEvolution)
    ) or not isinstance(
        dynamics.observation_model, (LinearGaussianObservation, GaussianObservation)
    ):
        raise TypeError(
            "gaussian_to_nlgssm_params expects a Gaussian DynamicalModel with "
            "state_evolution either LinearGaussianStateEvolution or "
            "GaussianStateEvolution, and observation_model either "
            "LinearGaussianObservation or GaussianObservation."
        )

    evo = dynamics.state_evolution
    obs = dynamics.observation_model
    ic = dynamics.initial_condition
    state_dim = dynamics.state_dim
    control_dim = dynamics.control_dim

    if isinstance(ic, dist.MultivariateNormal):
        initial_mean = jnp.asarray(ic.loc)
        initial_covariance = jnp.asarray(ic.covariance_matrix)
    elif isinstance(ic, dist.Normal):
        # dist.Normal: scalar Gaussian, treat as 1D state with variance scale^2.
        initial_mean = jnp.atleast_1d(jnp.asarray(ic.loc))
        initial_covariance = jnp.atleast_2d(jnp.square(jnp.asarray(ic.scale)))
    else:
        raise TypeError(
            "KF, EKF, and UKF require a Gaussian initial condition "
            "(MultivariateNormal or Normal) because they propagate mean and covariance. "
            "For non-Gaussian initial conditions, use filter_type='pf' (particle filter)."
        )

    # ----- Dynamics function -----
    if isinstance(evo, LinearGaussianStateEvolution):
        F = evo.A
        b = evo.bias if evo.bias is not None else jnp.zeros(state_dim)
        B = evo.B if evo.B is not None else jnp.zeros((state_dim, control_dim))

        def dynamics_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            out = F @ x + b
            if control_dim > 0 and u.size > 0:
                out = out + (B @ jnp.reshape(u, (-1, 1))).ravel()
            return out

    else:
        # GaussianStateEvolution: arbitrary nonlinear F(x, u, t_now, t_next) with Gaussian noise.
        def dynamics_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            # Discrete-time setting for EKF/UKF: ignore absolute time and pass dummy times.
            t_now = jnp.array(0.0, dtype=x.dtype)
            t_next = jnp.array(0.0, dtype=x.dtype)
            return evo.F(x, u, t_now, t_next)

    # ----- Emission function -----
    if isinstance(obs, LinearGaussianObservation):
        H = obs.H
        d = obs.bias if obs.bias is not None else jnp.zeros(dynamics.observation_dim)
        D = (
            obs.D
            if obs.D is not None
            else jnp.zeros((dynamics.observation_dim, control_dim))
        )

        def emission_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            out = H @ x + d
            if control_dim > 0 and u.size > 0:
                out = out + (D @ jnp.reshape(u, (-1, 1))).ravel()
            return out

    else:
        # GaussianObservation: y_t ~ N(h(x_t, u_t, t), R) with arbitrary h.
        def emission_function(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
            _t = jnp.array(0.0, dtype=x.dtype)
            return obs.h(x, u, _t)  # warning: time is ignored

    return ParamsNLGSSM(
        initial_mean=initial_mean,
        initial_covariance=initial_covariance,
        dynamics_function=dynamics_function,
        dynamics_covariance=evo.cov,
        emission_function=emission_function,
        emission_covariance=obs.R,
    )
