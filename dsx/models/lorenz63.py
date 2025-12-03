import jax
import jax.numpy as jnp
from dsx.dynamical_models import DynamicalModel
from dsx.observations import LinearGaussianObservation
from dsx.dynamical_models import ContinuousTimeStateEvolution, StochasticContinuousTimeStateEvolution
from dsx.dynamical_models import State, dState, Control, Time
from typing import Optional
from numpyro import distributions as dist
import dataclasses

# ----------------------------------------------------------------------
# Lorenz-63 ODE and SDE Implementations
# ----------------------------------------------------------------------

@dataclasses.dataclass
class Lorenz63ODE(ContinuousTimeStateEvolution):
    """
    Classic Lorenz-63 system:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    State is a dict: {'x': jnp.array([x, y, z])}
    Params require keys: {'sigma', 'rho', 'beta'}
    """
    
    rho: float = 28.0
    sigma: float = 10.0
    beta: float = 8.0 / 3.0

    def drift(self,
              x: State,
              u: Optional[Control],
              t: Time) -> dState:
        x, y, z = x[..., 0], x[..., 1], x[..., 2]

        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z

        # return {"x": jnp.stack([dx, dy, dz], axis=-1)}
        return jnp.array([dx, dy, dz])  # shape (..., 3)

@dataclasses.dataclass
class Lorenz63SDE(StochasticContinuousTimeStateEvolution, Lorenz63ODE):
    """
    Lorenz-63 SDE:
        dx = f(x,t;params) dt + L(x,t;params) dW_t

    Inherits drift() from Lorenz63ODE.

    diffusion_scale: scalar or array (broadcastable to (3,))
    Diffusion is isotropic by default:
        L = diag([sigma_x, sigma_y, sigma_z])
    """

    rho: float = 28.0
    sigma: float = 10.0
    beta: float = 8.0 / 3.0
    diffusion_scale: float = 1.0

    # -------------------------------------------------------
    # Diffusion coefficient (L)
    # -------------------------------------------------------
    def diffusion_coefficient(self,
        x: State,
        u: Optional[Control],
        t: Time) -> jax.Array:
        """
        Return the diffusion coefficient matrix L for the SDE:
            dx = f(x,t) dt + L dW_t

        Expected params:
            params["diffusion_scale"]: scalar or array_like of length 3.

        """
        scale = self.diffusion_scale

        # Convert to array and broadcast to shape (3,)
        scale = jnp.asarray(scale)
        if scale.ndim == 0:
            scale = jnp.ones((3,)) * scale
        else:
            scale = jnp.broadcast_to(scale, (3,))

        L = jnp.diag(scale) # shape (3, 3)

        return L


    # -------------------------------------------------------
    # Diffusion covariance (Q) as Identity
    # -------------------------------------------------------
    def diffusion_covariance(self,
        x: State,
        u: Optional[Control],
        t: Time) -> jax.Array:
        """ Assume dim_x-dimensional brownian motion with identity covariance."""
        return jnp.eye(3)


def make_L63_SDE_model(
    ic_mean=jnp.zeros(3),
    ic_cov=20**2*jnp.eye(3),
    obs_noise_sd=5.0,
    rho=28.0,
    sigma=10.0,
    beta=8/3,
    diffusion_scale=1.0,
):
    """
    Build a complete DynamicalModel for Lorenz-63 SDE with:
        - Gaussian IC
        - Lorenz-63 SDE dynamics
        - First-component Gaussian observation
    """
    
    def state_evolution():
        return Lorenz63SDE(rho=rho, sigma=sigma, beta=beta, diffusion_scale=diffusion_scale)
    
    def observation():
        H = jnp.array([[1.0, 0.0, 0.0]])  # observe first component
        R = jnp.array([[obs_noise_sd**2]]) # observation noise covariance
        return LinearGaussianObservation(H=H, R=R)

    def initial_condition():
        return dist.MultivariateNormal(loc=ic_mean, covariance_matrix=ic_cov)

    return DynamicalModel(
        initial_condition=initial_condition(),
        state_evolution=state_evolution(),
        observation_model=observation(),
        state_dim=3,
        observation_dim=1,
    )
