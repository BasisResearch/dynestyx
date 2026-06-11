"""State evolution implementations.

Specialty implementations for discrete-time systems. Structure allows future
extension to LTI factories, Neural SDEs, etc.
"""

from collections.abc import Callable
from typing import NamedTuple, cast

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, Real

from dynestyx.models.core import DiscreteTimeStateEvolution


class LinearGaussianParams(NamedTuple):
    """Linear-Gaussian transition parameters resolved at one time interval.

    Returned by `LinearGaussianStateEvolution.params_at`: any callable
    (time-varying) parameter has been evaluated at the requested interval, so
    every entry is a plain array (or `None` for an absent optional term).

    Expected shapes match the `LinearGaussianStateEvolution` fields; they are
    deliberately not enforced here because plate slicing can legally hand a
    member-sliced (reduced-rank) parameter to `__call__`.
    """

    A: Float[Array, "..."]
    B: Float[Array, "..."] | None
    bias: Float[Array, "..."] | None
    cov: Float[Array, "..."]


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

    Each parameter may be a constant array (time-invariant) or a callable
    `(t_now, t_next) -> value` evaluated per transition interval
    (time-varying); constant and callable parameters may be mixed freely.

    Note:
        - Callable parameters receive only the interval endpoints
          `(t_now, t_next)`; they must not depend on state or controls (use
          `GaussianStateEvolution` for nonlinear transitions).
        - Callables must be pure, JAX-traceable functions returning a fixed
          shape.
        - Backend support: time-varying parameters work with the simulators
          and the `filter_source="cuthbert"` filters/smoothers; the
          cd_dynamax backend requires constant arrays and raises `TypeError`
          otherwise.
    """

    A: (
        Float[Array, "*a_plate state_dim state_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*a_plate state_dim state_dim"],
        ]
    )
    cov: (
        Float[Array, "*cov_plate state_dim state_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*cov_plate state_dim state_dim"],
        ]
    )
    B: (
        Float[Array, "*b_matrix_plate state_dim control_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*b_matrix_plate state_dim control_dim"],
        ]
        | None
    ) = None
    bias: (
        Float[Array, "*bias_plate state_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*bias_plate state_dim"],
        ]
        | None
    ) = None

    def __init__(
        self,
        A: Float[Array, "*a_plate state_dim state_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*a_plate state_dim state_dim"],
        ],
        cov: Float[Array, "*cov_plate state_dim state_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*cov_plate state_dim state_dim"],
        ],
        B: Float[Array, "*b_matrix_plate state_dim control_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*b_matrix_plate state_dim control_dim"],
        ]
        | None = None,
        bias: Float[Array, "*bias_plate state_dim"]
        | Callable[
            [float | int | Real[Array, ""], float | int | Real[Array, ""]],
            Float[Array, "*bias_plate state_dim"],
        ]
        | None = None,
    ):
        """
        Args:
            A (jax.Array | Callable): State transition matrix with shape
                $(d_x, d_x)$, or a callable `(t_now, t_next)` returning it.
            cov (jax.Array | Callable): Process-noise covariance with shape
                $(d_x, d_x)$, or a callable `(t_now, t_next)` returning it.
            B (jax.Array | Callable | None): Optional control matrix with
                shape $(d_x, d_u)$, or a callable `(t_now, t_next)`
                returning it.
            bias (jax.Array | Callable | None): Optional additive bias with
                shape $(d_x,)$, or a callable `(t_now, t_next)` returning it.
        """
        self.A = A
        self.B = B
        self.bias = bias
        self.cov = cov

    @property
    def is_time_invariant(self) -> bool:
        """True iff every parameter is a constant array (no callables)."""
        return not any(
            callable(field) for field in (self.A, self.B, self.bias, self.cov)
        )

    def params_at(
        self,
        t_now: float | int | Real[Array, ""],
        t_next: float | int | Real[Array, ""],
    ) -> LinearGaussianParams:
        """Resolve `(A, B, bias, cov)` at one transition interval.

        Constant parameters are returned unchanged; callable parameters are
        evaluated at `(t_now, t_next)`.
        """

        def _resolve(field):
            if field is None or not callable(field):
                return field
            fn = cast(
                Callable[
                    [
                        float | int | Real[Array, ""],
                        float | int | Real[Array, ""],
                    ],
                    Array,
                ],
                field,
            )
            return jnp.asarray(fn(t_now, t_next))

        return LinearGaussianParams(
            A=_resolve(self.A),
            B=_resolve(self.B),
            bias=_resolve(self.bias),
            cov=_resolve(self.cov),
        )

    def __call__(self, x, u, t_now, t_next):
        A, B, bias, cov = self.params_at(t_now, t_next)
        loc = jnp.dot(A, x)
        if bias is not None:
            loc = loc + bias
        if B is not None and u is not None:
            loc = loc + jnp.dot(B, u)

        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)


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
