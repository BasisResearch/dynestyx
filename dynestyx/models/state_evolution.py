"""State evolution implementations.

Specialty implementations for discrete-time systems. Structure allows future
extension to LTI factories, Neural SDEs, etc.
"""

from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, Real

from dynestyx.models.core import DiscreteTimeStateEvolution


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

    A: Float[Array, "*a_plate state_dim state_dim"]
    cov: Float[Array, "*cov_plate state_dim state_dim"]
    B: Float[Array, "*b_matrix_plate state_dim control_dim"] | None = None
    bias: Float[Array, "*bias_plate state_dim"] | None = None

    def __init__(
        self,
        A: Float[Array, "*a_plate state_dim state_dim"],
        cov: Float[Array, "*cov_plate state_dim state_dim"],
        B: Float[Array, "*b_matrix_plate state_dim control_dim"] | None = None,
        bias: Float[Array, "*bias_plate state_dim"] | None = None,
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

    def __call__(self, x, u, t_now, t_next):
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
    process-noise covariance (either constant or state/time dependent).
    """

    F: Callable[
        [
            Real[Array, " state_dim"] | Real[Array, ""],
            Real[Array, " control_dim"] | Real[Array, ""] | None,
            float | int | Real[Array, ""],
            float | int | Real[Array, ""],
        ],
        Real[Array, " state_dim"] | Real[Array, ""],
    ]
    cov: (
        Float[Array, "*plate state_dim state_dim"]
        | Callable[
            [
                Real[Array, " state_dim"] | Real[Array, ""],
                Real[Array, " control_dim"] | Real[Array, ""] | None,
                float | int | Real[Array, ""],
                float | int | Real[Array, ""],
            ],
            Float[Array, "*plate state_dim state_dim"],
        ]
    )

    def __init__(
        self,
        F: Callable[
            [
                Real[Array, " state_dim"] | Real[Array, ""],
                Real[Array, " control_dim"] | Real[Array, ""] | None,
                float | int | Real[Array, ""],
                float | int | Real[Array, ""],
            ],
            Real[Array, " state_dim"] | Real[Array, ""],
        ],
        cov: Float[Array, "*plate state_dim state_dim"]
        | Callable[
            [
                Real[Array, " state_dim"] | Real[Array, ""],
                Real[Array, " control_dim"] | Real[Array, ""] | None,
                float | int | Real[Array, ""],
                float | int | Real[Array, ""],
            ],
            Float[Array, "*plate state_dim state_dim"],
        ],
    ):
        """
        Args:
            F (Callable[[State, Control, Time, Time], State]): Transition
                function mapping $(x, u, t_k, t_{k+1})$ to the conditional mean.
            cov (jax.Array | Callable[[State, Control, Time, Time], jax.Array]):
                Process-noise covariance with shape $(d_x, d_x)$, or a callable
                mapping $(x, u, t_k, t_{k+1})$ to that covariance.
        """
        self.F = F
        self.cov = cov

    def __call__(self, x, u, t_now, t_next):
        loc = self.F(x, u, t_now, t_next)
        if callable(self.cov):
            cov_fn = cast(
                Callable[
                    [
                        Real[Array, " state_dim"] | Real[Array, ""],
                        Real[Array, " control_dim"] | Real[Array, ""] | None,
                        float | int | Real[Array, ""],
                        float | int | Real[Array, ""],
                    ],
                    Float[Array, "*plate state_dim state_dim"],
                ],
                self.cov,
            )
            cov = cov_fn(x, u, t_now, t_next)
        else:
            cov = self.cov

        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)


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

    A: Float[Array, "*a_plate state_dim state_dim"]
    B: Float[Array, "*b_matrix_plate state_dim control_dim"] | None = None
    b: Float[Array, "*bias_plate state_dim"] | None = None

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"]:
        out = jnp.dot(self.A, x)
        if self.B is not None:
            u_vec = u if u is not None else jnp.zeros(self.B.shape[1])
            out = out + jnp.dot(self.B, u_vec)
        if self.b is not None:
            out = out + self.b
        return out
