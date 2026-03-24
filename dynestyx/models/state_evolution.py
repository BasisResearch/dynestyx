"""State evolution implementations.

Specialty implementations for discrete-time systems. Structure allows future
extension to LTI factories, Neural SDEs, etc.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Float

from dynestyx.models.core import DiscreteTimeStateEvolution
from dynestyx.types import Control, State, Time


class LinearGaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    Linear-Gaussian discrete-time state transition.

    The next state is modeled as

    $$
    x_{t_{k+1}} \\sim \\mathcal{N}(A x_{t_k} + B u_{t_k} + b, Q),
    $$

    where $A$ is the state transition matrix, $B$ is an optional control-input
    matrix, $b$ is an optional transition bias, and $Q$ is the process-noise
    covariance.
    """

    A: Float[jax.Array, "d_x d_x"]
    B: Float[jax.Array, "d_x d_u"] | None = None
    bias: Float[jax.Array, " d_x"] | None = None
    cov: Float[jax.Array, "d_x d_x"]

    def __init__(
        self,
        A: Float[jax.Array, "d_x d_x"],
        cov: Float[jax.Array, "d_x d_x"],
        B: Float[jax.Array, "d_x d_u"] | None = None,
        bias: Float[jax.Array, " d_x"] | None = None,
    ):
        """
        Args:
            A (jax.Array): State transition matrix with shape
                $(d_x, d_x)$.
            cov (jax.Array): Process-noise covariance with shape
                $(d_x, d_x)$.
            B (jax.Array | None): Optional control matrix with shape
                $(d_x, d_u)$.
            bias (jax.Array | None): Optional additive bias with shape
                $(d_x,)$.
        """
        self.A = A
        self.B = B
        self.bias = bias
        self.cov = cov

    def __call__(
        self,
        x: Float[jax.Array, " d_x"],
        u: Float[jax.Array, " d_u"] | None,
        t_now: Float[jax.Array, ""],
        t_next: Float[jax.Array, ""],
    ) -> dist.MultivariateNormal:
        loc = jnp.dot(self.A, x)
        if self.bias is not None:
            loc += self.bias
        if self.B is not None and u is not None:
            loc += jnp.dot(self.B, u)

        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov)


class GaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    Nonlinear Gaussian discrete-time state transition.

    The next state is modeled as

    $$
    x_{t_{k+1}} \\sim \\mathcal{N}(F(x_{t_k}, u_{t_k}, t_k, t_{k+1}), Q),
    $$

    where $F$ is a user-provided transition function and $Q$ is the
    process-noise covariance.
    """

    F: Callable[[State, Control, Time, Time], State]
    cov: Float[jax.Array, "d_x d_x"]

    def __init__(
        self,
        F: Callable[[State, Control, Time, Time], State],
        cov: Float[jax.Array, "d_x d_x"],
    ):
        """
        Args:
            F (Callable[[State, Control, Time, Time], State]): Transition
                function mapping $(x, u, t_k, t_{k+1})$ to the conditional mean.
            cov (jax.Array): Process-noise covariance with shape
                $(d_x, d_x)$.
        """
        self.F = F
        self.cov = cov

    def __call__(
        self,
        x: Float[jax.Array, " d_x"],
        u: Float[jax.Array, " d_u"] | None,
        t_now: Float[jax.Array, ""],
        t_next: Float[jax.Array, ""],
    ) -> dist.MultivariateNormal:
        loc = self.F(x, u, t_now, t_next)

        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov)


class AffineDrift(eqx.Module):
    """
    Affine drift function for continuous-time models.

    This implements an affine map of the form

    $$f(x, u, t) = A x + B u + b,$$

    where $A \\in \\mathbb{R}^{d_x \\times d_x}$, $B \\in \\mathbb{R}^{d_x \\times d_u}$
    (optional), and $b \\in \\mathbb{R}^{d_x}$ (optional). The time argument $t$
    is accepted for compatibility with the `Drift` protocol but is not used.

    This is commonly used as the drift term $\\mu(x_t, u_t, t)$ inside
    `ContinuousTimeStateEvolution`, and is a building block for LTI models such as
    `LTI_continuous`.

    Attributes:
        A (jax.Array): Drift matrix with shape $(d_x, d_x)$.
        B (jax.Array | None): Optional control matrix with shape $(d_x, d_u)$.
        b (jax.Array | None): Optional additive bias with shape $(d_x,)$.
    """

    A: Float[jax.Array, "d_x d_x"]
    B: Float[jax.Array, "d_x d_u"] | None = None
    b: Float[jax.Array, " d_x"] | None = None

    def __call__(
        self,
        x: Float[jax.Array, " d_x"],
        u: Float[jax.Array, " d_u"] | None,
        t: Float[jax.Array, ""],
    ) -> Float[jax.Array, " d_x"]:
        out = jnp.dot(self.A, x)
        if self.B is not None:
            u_vec = u if u is not None else jnp.zeros(self.B.shape[1])
            out = out + jnp.dot(self.B, u_vec)
        if self.b is not None:
            out = out + self.b
        return out
