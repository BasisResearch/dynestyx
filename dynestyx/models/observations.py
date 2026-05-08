"""Observation model implementations."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from numpyro import distributions as dist

from dynestyx.models.core import ObservationModel
from dynestyx.types import Control, Observation, State, Time


class LinearGaussianObservation(ObservationModel):
    """
    Linear-Gaussian observation model.

    Observations are modeled as

    $$
    y_t \\sim \\mathcal{N}(H x_t + D u_t + b, R).
    $$

    Here, $H$ is the observation matrix, $D$ is an optional control-input
    matrix, $b$ is an optional observation bias, and $R$ is the observation
    noise covariance.
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
        Args:
            H (jax.Array): Observation matrix with shape
                $(d_y, d_x)$.
            R (jax.Array): Observation noise covariance with shape
                $(d_y, d_y)$.
            D (jax.Array | None): Optional control matrix with shape
                $(d_y, d_u)$. If None, no control contribution is used.
            bias (jax.Array | None): Optional additive bias with shape
                $(d_y,)$.
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
    Nonlinear Gaussian observation model.

    Observations are modeled as

    $$
    y_t \\sim \\mathcal{N}(h(x_t, u_t, t), R),
    $$

    where $h$ is a user-provided measurement function and $R$ is the
    observation noise covariance.
    """

    h: Callable[[State, Control, Time], Observation]
    R: jax.Array

    def __init__(self, h: Callable[[State, Control, Time], jax.Array], R: jax.Array):
        """
        Args:
            h (Callable[[State, Control, Time], jax.Array]): Measurement
                function mapping $(x, u, t)$ to the mean observation.
            R (jax.Array): Observation noise covariance with shape
                $(d_y, d_y)$.
        """
        self.h = h
        self.R = R

    def __call__(self, x, u, t):
        loc = self.h(x, u, t)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.R)


class DiracIdentityObservation(ObservationModel):
    """
    Noise-free identity observation model.

    Observations are modeled as

    $$
    y_t \\sim \\delta(x_t),
    $$

    i.e., the observation equals the latent state almost surely.
    """

    def __call__(self, x, u, t):
        # Treat scalar latent states as scalar events, and otherwise use only
        # the trailing state axis as the event dimension so any leading batch
        # or plate axes are preserved.
        event_dim = 0 if jnp.ndim(x) == 0 else 1
        return dist.Delta(x, event_dim=event_dim)
