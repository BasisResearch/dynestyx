from collections.abc import Callable

import jax
import jax.numpy as jnp
from numpyro import distributions as dist

from dynestyx.dynamical_models import Control, ObservationModel, State, Time


class LinearGaussianObservation(ObservationModel):
    """
    y_t | x_t, u_t, t ~ Normal( H x_t + D u_t + bias, R )

    where H is the observation matrix, D is the control matrix, and R is the observation noise covariance.
    """

    H: jax.Array
    R: jax.Array
    D: jax.Array | None = None
    bias: jax.Array | None = None

    def __init__(
        self,
        H: jax.Array,
        R: jax.Array,
        D: jax.Array | None = None,
        bias: jax.Array | None = None,
    ):
        """
        H: Observation matrix, shape (obs_dim, state_dim)
        R: Observation noise covariance, shape (obs_dim, obs_dim)
        """
        self.H = H
        self.D = D
        self.R = R
        self.bias = bias

    def __call__(self, x, u, t):
        loc = jnp.dot(self.H, x)
        if self.D is not None and u is not None:
            loc += jnp.dot(self.D, u)
        if self.bias is not None:
            loc += self.bias
        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.R)


class GaussianObservation(ObservationModel):
    """
    y_t | x_t, u_t, t ~ Normal(h(x_t, u_t, t), R)
    where h is a callable mapping (State, Control, Time) -> State
    and R is the observation noise covariance.
    """

    h: Callable[[State, Control, Time], jax.Array]
    R: jax.Array

    def __init__(self, h: Callable[[State, Control, Time], jax.Array], R: jax.Array):
        self.h = h
        self.R = R

    def __call__(self, x, u, t):
        loc = self.h(x, u, t)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.R)


class DiracIdentityObservation(ObservationModel):
    """
    y_t | x_t ~ DiracDelta(x_t)
    """

    def __call__(self, x, u, t):
        return dist.Delta(x)
