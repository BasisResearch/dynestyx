import jax.numpy as jnp
from typing import Optional, Tuple
from jax import Array

from dsx.dynamical_models import DynamicalModel, ContinuousTimeStateEvolution
from dsx.observations import LinearGaussianObservation
from dsx.ops import Context
from numpyro import distributions as dist
import numpyro
from numpyro.primitives import _PYRO_STACK, CondIndepStackFrame

from cd_dynamax import ContDiscreteNonlinearGaussianSSM as CDNLGSSM
from cd_dynamax import ContDiscreteNonlinearSSM as CDNLSSM

from typing import TypeAlias

import jax.random as jr
import diffrax as dfx

SSMType: TypeAlias = CDNLGSSM | CDNLSSM


def infer_batch_shape():
    """Infer the current plate-induced batch shape, or None if not in a plate."""
    cond_indep_stack = []
    for frame in _PYRO_STACK:
        if isinstance(frame, numpyro.primitives.plate):
            cond_indep_stack.append(
                CondIndepStackFrame(frame.name, frame.dim, frame.size)
            )
    if cond_indep_stack:
        return numpyro.primitives.plate._get_batch_shape(cond_indep_stack)
    return None


def dsx_to_cd_dynamax(
    dsx_model: DynamicalModel, cd_model: Optional[SSMType] = None
) -> Tuple[dict, bool]:
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
            model_to_use: SSMType = CDNLSSM(
                state_dim=dsx_model.state_dim,
                emission_dim=dsx_model.observation_dim,
                input_dim=dsx_model.control_dim,
            )
        else:
            model_to_use = CDNLGSSM(
                state_dim=dsx_model.state_dim,
                emission_dim=dsx_model.observation_dim,
                input_dim=dsx_model.control_dim,
            )
    else:
        model_to_use = cd_model

    cd_dynamax_params = model_to_use.build_params(**params)

    return cd_dynamax_params, non_gaussian_flag


def _validate_control_dim(
    dynamics: DynamicalModel, ctrl_values: Optional[Array]
) -> None:
    """
    Validate that control_dim is set in DynamicalModel when controls are present.

    Args:
        dynamics: DynamicalModel instance
        ctrl_values: Control values array or None

    Raises:
        ValueError: If controls are provided but control_dim is not set or is 0
    """
    if ctrl_values is not None:
        if dynamics.control_dim is None or dynamics.control_dim == 0:
            # Try to infer from shape
            if ctrl_values.ndim >= 2:
                inferred_dim = ctrl_values.shape[1]
                raise ValueError(
                    f"Controls are provided (shape: {ctrl_values.shape}), but "
                    f"dynamics.control_dim is {dynamics.control_dim}. "
                    f"Please set control_dim={inferred_dim} when creating the DynamicalModel."
                )
            else:
                raise ValueError(
                    f"Controls are provided, but dynamics.control_dim is {dynamics.control_dim}. "
                    "Please set control_dim when creating the DynamicalModel."
                )


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

    if ctrl_times is None:
        if ctrl_traj.values is not None:
            raise ValueError(
                "ctrl_traj.values is not None, but ctrl_times is None. This is likely a bug in the context creation."
            )
        # No controls provided
        return None, None
    elif ctrl_traj.values is None:
        raise ValueError(
            "ctrl_traj.values is None, but ctrl_times is not None. This is likely a bug in the context creation."
        )

    # Check lengths match (concrete check, safe in traced context)
    if len(ctrl_times) != len(obs_times):
        raise ValueError(
            f"Control times length ({len(ctrl_times)}) must match "
            f"observation times length ({len(obs_times)})"
        )
    # Note: Full equality check would require jnp.array_equal which creates
    # traced booleans. We trust that if lengths match, times match (validated
    # at fixture/context creation time).

    # Controls are provided (have times), extract and validate
    ctrl_values = ctrl_traj.values

    # Validate ctrl_values is not a dict
    if isinstance(ctrl_values, dict):
        raise ValueError("ctrl_values must be an Array or None, not a dict")

    return ctrl_times, ctrl_values


def _get_val_or_None(values: Optional[Array], t_idx: int) -> Optional[Array]:
    """
    Safely get value at index t_idx, returning None if values is None.

    Args:
        values: Values array or None
        t_idx: Time index to access

    Returns:
        Value at index t_idx, or None if values is None
    """
    return values[t_idx] if values is not None else None


def diffeqsolve_util(
    drift,
    t0: float,
    t1: float,
    y0: jnp.ndarray,
    reverse: bool = False,
    args=None,
    solver: dfx.AbstractSolver = None,
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
    dt0: float = 0.01,
    tol_vbt: float = 1e-1,  # tolerance for virtual brownian tree
    max_steps: int = 1e5,
    diffusion=None,
    key=None,
    debug=False,
    **kwargs,
) -> jnp.ndarray:
    """
    The CD-dynamax [1] wrapper for solving differential equations, with some automatic default choices.

    Choosing solvers and adjoints based on diffrax website's recommendation for training neural ODEs.
        See: https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/
        See: https://docs.kidger.site/diffrax/api/adjoints/

        Note that choosing RecursionCheckpointAdjoint requires usage of reverse-mode auto-differentiation.
        Can use DirectAdjoint for flexible forward-mode + reverse-mode auto-differentiation.

        Defaults are chosen to be decent low-cost options for forward solves and backpropagated gradients.

        If you want high-fidelity solutions (and their gradients), it is recommended
        - for ODEs: choose a higher-order solver (Tsit5) and an adaptive stepsize controller (PIDController).
        - for SDEs: follow diffrax website advice (definitely can choose dt0 very small with constant stepsize controller).

        Things to pay attention to (that we have incomplete understanding of):
        - checkpoints in RecursiveCheckpointAdjoint: this is used to save memory during backpropagation.
        - max_steps: reducing this can speed things up. But it is also used to set default number of checkpoints.
        ... unclear the optimal way to set these parameters.

    [1] https://github.com/hd-UQ/cd_dynamax/blob/633373e11322a4f875cfb0ad7a1817f82ad6787b/cd_dynamax/src/utils/diffrax_utils.py
    """

    max_steps = int(max_steps)

    if debug:
        # run hand-written Euler and/or Euler-Maruyama using a for loop with fixed step size dt0
        N = 200  # if this is too small, then the error will be too large and covariances can be very non-SPD.
        dt = (t1 - t0) / N
        if key is None:
            key = jr.PRNGKey(0)
        keys = jr.split(key, N)

        for i in range(N):
            drift_i = drift(t0 + i * dt, y0, None)
            if isinstance(y0, tuple):
                # If y0 and drift_i are tuples, update each component
                y0 = tuple(
                    y0_component + dt * drift_component
                    for y0_component, drift_component in zip(y0, drift_i)
                )
                if diffusion is not None:
                    diff = diffusion(t0 + i * dt, y0, None)
                    rnd = tuple(
                        jr.normal(key=keys[i], shape=y0_component.shape)
                        for y0_component in y0
                    )
                    y0 = tuple(
                        y0_component + jnp.sqrt(dt) * diff_component * rnd_component
                        for y0_component, diff_component, rnd_component in zip(
                            y0, diff, rnd
                        )
                    )
            else:
                # If y0 and drift_i are vectors, update directly
                y0 = y0 + dt * drift_i
                if diffusion is not None:
                    diff = diffusion(t0 + i * dt, y0, None)
                    rnd = jr.normal(key=keys[i], shape=y0.shape)
                    y0 = y0 + jnp.sqrt(dt) * diff * rnd

        # return the final state y0 with an additional first dimension
        if isinstance(y0, tuple):
            # Reshape to match the expected output of the solver
            return tuple(jnp.expand_dims(y0_component, axis=0) for y0_component in y0)
        else:
            return y0

    # set solver to default if not provided
    if solver is None:
        if diffusion is None:
            solver = dfx.Dopri5()
            # Tsit5 may be another slightly better default method.
        else:
            solver = dfx.Heun()
            # sometimes called the improved Euler method

    # allow for reverse-time integration
    # if t1 < t0, we assume that initial condition y0 is at t1
    if reverse:
        t0_new = 0
        t1_new = t1 - t0
        drift_new = reverse_rhs(drift, t1, y0)
        diffusion_new = reverse_rhs(diffusion, t1, y0)
    else:
        t0_new = t0
        t1_new = t1
        drift_new = drift
        diffusion_new = diffusion

    # set DE terms
    if diffusion_new is None:
        terms = dfx.ODETerm(drift_new)
    else:
        # Important: `shape` here specifies the shape of the Brownian increment (control),
        # not the shape of the state.
        #
        # If `y0` is batched (e.g. shape (B, state_dim)), setting `shape=y0.shape` would
        # create a batched Brownian control and can cause shape mismatches inside diffrax
        # (e.g. when contracting diffusion with the control).
        #
        # We instead use only the trailing (state) dimension(s) as the Brownian shape.
        bm_shape = y0.shape[-1:] if hasattr(y0, "shape") else ()
        bm = dfx.VirtualBrownianTree(
            t0=t0_new, t1=t1_new, tol=tol_vbt, shape=bm_shape, key=key
        )
        terms = dfx.MultiTerm(
            dfx.ODETerm(drift_new), dfx.ControlTerm(diffusion_new, bm)
        )

    # return a specific solver
    sol = dfx.diffeqsolve(
        terms,
        solver=solver,
        stepsize_controller=stepsize_controller,
        t0=t0_new,
        t1=t1_new,
        y0=y0,
        args=args,
        dt0=dt0,
        saveat=dfx.SaveAt(t1=True),
        adjoint=adjoint,
        max_steps=max_steps,
        **kwargs,
    ).ys

    return sol


def reverse_rhs(rhs, t1, ref_var):
    """
    Utility from CD-dynamax [1] for a time-reversed right-hand-side of a differential equation.

    [1] https://github.com/hd-UQ/cd_dynamax/blob/633373e11322a4f875cfb0ad7a1817f82ad6787b/cd_dynamax/src/utils/diffrax_utils.py
    """
    if rhs is None:
        return None

    if isinstance(ref_var, tuple):

        def rev_rhs(s, y, args):
            foo = rhs(t1 - s, y, args)
            return tuple(-f for f in foo)
    else:

        def rev_rhs(s, y, args):
            return -rhs(t1 - s, y, args)

    return rev_rhs
