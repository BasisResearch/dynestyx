"""State evolution implementations.

Specialty implementations for discrete-time systems. Structure allows future
extension to LTI factories, Neural SDEs, etc.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx.models.core import DiscreteTimeStateEvolution
from dynestyx.types import Control, State, Time, dState


class LinearGaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    x_t_next | x_t_now, u_t_now, t_now, t_next ~ Normal( A x_t_now + B u_t_now + bias, cov )

    where A is the observation matrix, B is the control matrix, bias is the bias, and cov is the state noise covariance.
    """

    A: jax.Array
    B: jax.Array | None = None
    bias: jax.Array | None = None
    cov: jax.Array

    def __init__(
        self,
        A: jax.Array,
        cov: jax.Array,
        B: jax.Array | None = None,
        bias: jax.Array | None = None,
    ):
        self.A = A
        self.B = B
        self.bias = bias
        self.cov = cov

    def __call__(self, x, u, t_now, t_next):
        loc = jnp.dot(self.A, x)
        if self.bias is not None:
            loc += self.bias
        if self.B is not None and u is not None:
            loc += jnp.dot(self.B, u)

        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov)


class GaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    x_t_next | x_t_now, u_t_now, t_now, t_next ~ Normal( F(x_t_now, u_t_now, t_now, t_next), cov )

    where F is a callable mapping (State, Control, Time) -> State
    and cov is the state noise covariance.
    """

    F: Callable[[State, Control, Time, Time], State]
    cov: jax.Array

    def __init__(
        self,
        F: Callable[[State, Control, Time, Time], State],
        cov: jax.Array,
    ):
        self.F = F
        self.cov = cov

    def __call__(self, x, u, t_now, t_next):
        loc = self.F(x, u, t_now, t_next)

        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov)


class AffineDrift(eqx.Module):
    """
    Affine drift: f(x, u, t) = A @ x + B @ u + b.
    """

    A: jax.Array
    B: jax.Array | None = None
    b: jax.Array | None = None

    def __call__(
        self,
        x: State,
        u: Control | None,
        t: Time,
    ) -> dState:
        out = jnp.dot(self.A, x)
        if self.B is not None:
            u_vec = u if u is not None else jnp.zeros(self.B.shape[1])
            out = out + jnp.dot(self.B, u_vec)
        if self.b is not None:
            out = out + self.b
        return out
