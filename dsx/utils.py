import jax.numpy as jnp
from typing import Optional, Tuple
from jax import Array

from dsx.dynamical_models import DynamicalModel, ContinuousTimeStateEvolution
from dsx.observations import LinearGaussianObservation
from dsx.ops import Context
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
    if isinstance(obs, LinearGaussianObservation):
        params.update(
            {
                "emission_function": lambda x, u, t: obs.H @ x,
                "emission_cov": obs.R,  # type: ignore
            }
        )
    else:
        # TODO: check for linear-gaussian observation models and extract H, R

        raise NotImplementedError(
            f"Observation model of type {type(obs)} is not supported yet."
        )

    cdnlgssm = CDNLGSSM(
        state_dim=dsx_model.state_dim, emission_dim=dsx_model.observation_dim
    )
    cd_dynamax_params = cdnlgssm.build_params(**params)

    return cd_dynamax_params


def _get_controls(
    context: Context, obs_times: Array
) -> Tuple[Optional[Array], Optional[Array]]:
    """
    Extract and validate controls from context.

    Args:
        context: Context containing controls trajectory
        obs_times: Observation times array for validation

    Returns:
        Tuple of (ctrl_times, ctrl_values). Both are None if no controls are provided.
        If controls are provided, ctrl_times and ctrl_values are extracted and validated.

    Raises:
        ValueError: If control times length doesn't match observation times length,
                    or if ctrl_values is a dict.
    """
    # Pull control trajectory from context
    # Only validate controls if they actually have times
    # If controls is a Trajectory with times=None, treat it as no controls
    ctrl_traj = context.controls
    ctrl_times = ctrl_traj.times if ctrl_traj is not None else None
    ctrl_values = ctrl_traj.values if ctrl_times is not None else None

    # If controls are provided (have times), verify that control times match observation times
    if ctrl_times is not None:
        # Check lengths match (concrete check, safe in traced context)
        if len(ctrl_times) != len(obs_times):
            raise ValueError(
                f"Control times length ({len(ctrl_times)}) must match "
                f"observation times length ({len(obs_times)})"
            )
        # Note: Full equality check would require jnp.array_equal which creates
        # traced booleans. We trust that if lengths match, times match (validated
        # at fixture/context creation time).
        if isinstance(ctrl_values, dict):
            raise ValueError("ctrl_values must be an Array or None, not a dict")

    return ctrl_times, ctrl_values
