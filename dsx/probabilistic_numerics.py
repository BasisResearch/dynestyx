"""Probabilistic numerics utilities for ODE solving with rodeo."""

import jax.numpy as jnp
from typing import Optional, Callable, Any, Tuple
import numpy as np

import rodeo
from rodeo.prior import ibm_init
from rodeo.utils import first_order_pad

from dsx.dynamical_models import DynamicalModel


def check_regular_grid(times: jnp.ndarray) -> float:
    """Check if times are on a regular grid and return the step size dt.
    
    Raises ValueError if times are not on a regular grid.
    Uses relaxed tolerances to account for floating-point precision.
    
    Note: This check is performed on concrete values to avoid JAX tracing issues.
    """
    if len(times) < 2:
        raise ValueError("Need at least 2 time points to check for regular grid")
    
    # Convert to numpy for the check to avoid JAX tracing issues
    # This works because times should be concrete (not traced) at this point
    times_np = np.asarray(times)
    
    # Compute differences between consecutive times
    dt_np = times_np[1] - times_np[0]
    diffs_np = np.diff(times_np)
    
    # Check if all differences are approximately equal (within numerical tolerance)
    # Use more relaxed tolerances for floating-point comparisons
    atol = 1e-6
    rtol = 1e-4
    if not np.allclose(diffs_np, dt_np, atol=atol, rtol=rtol):
        raise ValueError(
            f"Times must be on a regular grid. "
            f"First dt={dt_np}, but differences vary: min={np.min(diffs_np)}, max={np.max(diffs_np)}"
        )
    
    return float(dt_np)


def get_default_prior_pars(dt: float, state_dim: int, sigma: float = 5e7) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Get default prior parameters using IBM prior.
    
    For first-order vector ODEs, n_deriv = n_vars = state_dim.
    This follows the rodeo convention for first-order systems.
    
    Args:
        dt: Time step size.
        state_dim: State dimension (number of variables).
        sigma: Prior variance parameter.
    
    Returns:
        Tuple of (prior_weight, prior_var) with shapes (state_dim, state_dim, state_dim).
    """
    # For first-order vector ODEs: n_deriv = n_vars = state_dim
    # One sigma per variable/block
    sigma_array = jnp.array([sigma] * state_dim)  # (state_dim,)
    prior_weight, prior_var = ibm_init(dt, state_dim, sigma_array)
    return prior_weight, prior_var


def prepare_rodeo_inputs(
    dynamics: DynamicalModel,
    obs_times: jnp.ndarray,
    x0: jnp.ndarray,
    ode_weight: Optional[jnp.ndarray] = None,
    interrogate: Optional[Callable] = None,
    prior_pars: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    sigma: float = 5e7,
    **ode_params,
) -> dict:
    """Prepare all inputs for rodeo.solve_mv.
    
    Follows the pattern from rodeo examples (e.g., lorenz.md).
    
    Args:
        dynamics: Dynamical model.
        obs_times: Observation times (must be on regular grid).
        x0: Initial state, shape (state_dim,).
        ode_weight: Optional weight matrix. If None, uses first_order_pad.
        interrogate: Optional interrogate function. If None, uses default.
        prior_pars: Optional prior parameters. If None, uses default.
        kalman_type: Type of Kalman filter.
        sigma: Prior variance parameter for default prior.
        **ode_params: Additional parameters to pass to ODE function.
    
    Returns:
        Dictionary with all parameters needed for rodeo.solve_mv.
    """
    # Check regular grid and get dt
    dt = check_regular_grid(obs_times)
    
    state_dim = dynamics.state_dim
    T = len(obs_times)
    n_deriv = state_dim  # For first-order ODEs: n_deriv = n_vars = state_dim
    n_vars = state_dim

    # Define ODE function matching rodeo convention (matching reference example)
    # The ODE function passed to first_order_pad receives x0 with shape (n_vars, 1)
    # But when called by rodeo during solving, it receives X_t with shape (n_vars, n_deriv)
    def ode_fun(X_t, t, **params):
        # X_t can be (n_vars, 1) from first_order_pad or (n_vars, n_deriv) from rodeo
        # Extract first column to get actual state
        if X_t.ndim == 2 and X_t.shape[1] > 1:
            x_state = X_t[:, 0]  # (n_vars,) - from rodeo during solving
        else:
            x_state = X_t[:, 0] if X_t.ndim == 2 else X_t  # Handle both cases
        drift = dynamics.state_evolution.drift(x=x_state, u=None, t=t)
        # Return shape (n_vars, 1) - drift for each variable
        return drift[:, None]  # (n_vars, 1)
    
    # Use first_order_pad to get W and init_pad function (matching reference example)
    if ode_weight is None:
        W, init_pad = first_order_pad(ode_fun, n_vars, n_deriv)
        # Pad initial condition using init_pad (x0 should be (n_vars,))
        if x0.ndim == 0:
            x0 = jnp.array([x0])
        if x0.ndim == 1:
            ode_init = init_pad(x0, 0, **ode_params)
        else:
            ode_init = x0
    else:
        W = ode_weight
        # Manual padding if W is provided
        if x0.ndim == 0:
            x0 = jnp.array([x0])
        if x0.ndim == 1:
            x0_padded = jnp.zeros((state_dim, state_dim))
            x0_padded = x0_padded.at[:, 0].set(x0)
            ode_init = x0_padded
        else:
            ode_init = x0
        
    if prior_pars is None:
        prior_pars = get_default_prior_pars(dt, state_dim, sigma)
    
    # Compute time range and steps
    t_min = obs_times[0]
    t_max = obs_times[-1]
    n_steps = T - 1  # rodeo returns n_steps+1 points
    
    return {
        "ode_fun": ode_fun,
        "ode_weight": W,
        "ode_init": ode_init,
        "t_min": t_min,
        "t_max": t_max,
        "n_steps": n_steps,
        "interrogate": interrogate,
        "prior_pars": prior_pars,
    }


def reshape_rodeo_output(
    mean_state_smooth: jnp.ndarray,
    var_state_smooth: jnp.ndarray,
    T: int,
    state_dim: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Reshape rodeo output to (T, state_dim) format.
    
    Args:
        mean_state_smooth: Rodeo output, shape (n_steps+1, n_block, n_bstate)
            For multi-block: (T, state_dim, 1) - one block per dimension
        var_state_smooth: Rodeo output, shape (T, n_block, n_bstate, n_bstate)
            For multi-block: (T, state_dim, 1, 1) or (T, state_dim, 2, 2) depending on prior
        T: Number of time points.
        state_dim: State dimension.
    
    Returns:
        Tuple of (states_mean, states_cov) with shapes (T, state_dim) and (T, state_dim, state_dim).
    """
    n_block = mean_state_smooth.shape[1]
    
    if n_block == state_dim:
        # Multi-block case: one block per dimension
        # mean_state_smooth is (T, state_dim, n_bstate) where n_bstate = state_dim (n_deriv)
        # For first-order ODEs with n_deriv=state_dim, extract the first component from each block
        states_mean = mean_state_smooth[:, :, 0]  # (T, state_dim)
        
        # For covariance, var_state_smooth is (T, state_dim, n_bstate, n_bstate)
        # where n_bstate = state_dim. Extract the (0,0) element from each block
        # which gives the variance of the first component (the actual state variable)
        var_diag = var_state_smooth[:, :, 0, 0]  # (T, state_dim) - diagonal variances
        # Build diagonal covariance matrices (assuming independence between dimensions)
        states_cov = jnp.array([jnp.diag(var_diag[t]) for t in range(T)])  # (T, state_dim, state_dim)
    elif n_block == 1:
        # Single block case: (T, 1, state_dim) -> (T, state_dim)
        states_mean = mean_state_smooth[:, 0, :]  # (T, state_dim)
        states_cov = var_state_smooth[:, 0, :, :]  # (T, state_dim, state_dim)
    else:
        raise ValueError(f"Unexpected n_block={n_block}, expected 1 or {state_dim}")
    
    return states_mean, states_cov


def build_joint_covariance(states_cov: jnp.ndarray, T: int, state_dim: int) -> jnp.ndarray:
    """Build block-diagonal covariance matrix from per-time covariances.
    
    Note: This assumes independence across time points, which is an approximation.
    The full joint covariance from rodeo would include cross-time correlations.
    
    Args:
        states_cov: Per-time covariances, shape (T, state_dim, state_dim).
        T: Number of time points.
        state_dim: State dimension.
    
    Returns:
        Block-diagonal covariance matrix, shape (T * state_dim, T * state_dim).
    """
    total_dim = T * state_dim
    covariance_flat = jnp.zeros((total_dim, total_dim))
    for t in range(T):
        start_idx = t * state_dim
        end_idx = (t + 1) * state_dim
        covariance_flat = covariance_flat.at[start_idx:end_idx, start_idx:end_idx].set(
            states_cov[t]
        )
    return covariance_flat
