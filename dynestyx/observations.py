import jax
import jax.numpy as jnp
from dynestyx.dynamical_models import ObservationModel
from numpyro import distributions as dist


class LinearGaussianObservation(ObservationModel):
    """
    y_t | x_t ~ Normal( H x_t, R )

    where H is the observation matrix and R is the observation noise covariance.
    """

    H: jax.Array
    R: jax.Array

    def __init__(self, H: jax.Array, R: jax.Array):
        """
        H: Observation matrix, shape (obs_dim, state_dim)
        R: Observation noise covariance, shape (obs_dim, obs_dim)
        """
        self.H = H
        self.R = R

    def __call__(self, x, u, t):
        return dist.MultivariateNormal(loc=jnp.dot(self.H, x), covariance_matrix=self.R)


class DiracIdentityObservation(ObservationModel):
    """
    y_t | x_t ~ DiracDelta(x_t)
    """

    def __call__(self, x, u, t):
        return dist.Delta(x)
