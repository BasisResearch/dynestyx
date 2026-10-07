"""State evolution implementations.

Specialty implementations for discrete-time systems. Structure allows future
extension to LTI factories, Neural SDEs, etc.
"""

import warnings
from collections.abc import Callable
from typing import NamedTuple, cast

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, Real

from dynestyx.distributions import (
    MixedStateDistribution,
    _extract_and_validate_mixed_state,
)
from dynestyx.models.core import DiscreteTimeStateEvolution
from dynestyx.models.drifts import AffineDrift as _AffineDrift


class AffineDrift(_AffineDrift):
    """Deprecated alias for `dynestyx.models.drifts.AffineDrift`.

    Deprecated: import `AffineDrift` from `dynestyx.models.drifts` (or
    `dynestyx`) instead. This alias will be removed in v0.5.0.
    """

    def __check_init__(self) -> None:
        warnings.warn(
            "`dynestyx.models.state_evolution.AffineDrift` is deprecated; "
            "import `AffineDrift` from `dynestyx.models.drifts` (or "
            "`dynestyx`) instead. This alias will be removed in v0.5.0.",
            DeprecationWarning,
            stacklevel=2,
        )


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


class SwitchingLinearGaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    Switching linear-Gaussian discrete-time transition for SLDS models.

    z_t | z_{t-1} ~ Categorical(P[z_{t-1}]) ## discrete switching latent
    x_t | x_{t-1}, z_t ~ Normal(A[z_t] x_{t-1} + B[z_t] u_t + b[z_t], Q[z_t]) ## continuous linear dynamics latent
    state = [z, x_0, ..., x_{D-1}]

    The full simulator state is represented as [z, x...], where z is a
    discrete regime encoded as the first scalar entry and `x` is the continuous
    latent state.
    """

    transition_matrix: Float[Array, "num_regimes num_regimes"]
    A: Float[Array, "num_regimes state_dim state_dim"]
    cov: Float[Array, "num_regimes state_dim state_dim"]
    B: Float[Array, "num_regimes state_dim control_dim"] | None = None
    bias: Float[Array, "num_regimes state_dim"] | None = None

    rounding: bool = False

    def __init__(
        self,
        transition_matrix,
        A,
        cov,
        B=None,
        bias=None,
        *,
        rounding: bool = False,
    ):
        """
        Args:
            transition_matrix: Regime transition matrix with shape `(K, K)`.
            A: Regime-specific dynamics matrices with shape `(K, D, D)`.
            cov: Regime-specific process covariances with shape `(K, D, D)`.
            B: Optional regime-specific control matrices with shape `(K, D, U)`.
            bias: Optional regime-specific dynamics biases with shape `(K, D)`.
            rounding: Round the mixed state discrete component to the nearest
                integer (ties to even) before validation. Defaults to False.
                Nonfinite or out-of-range results raise an error.
        """
        self.transition_matrix = transition_matrix
        self.A = A
        self.cov = cov
        self.B = B
        self.bias = bias
        self.rounding = rounding

    @property
    def num_regimes(self) -> int:
        return int(self.transition_matrix.shape[0])

    @property
    def continuous_state_dim(self) -> int:
        return int(self.A.shape[-1])

    def __call__(self, x, u, t_now, t_next):
        z, x_cont, valid = _extract_and_validate_mixed_state(
            x, self.num_regimes, rounding=self.rounding
        )
        z = eqx.error_if(
            z,
            ~valid,
            "Mixed state discrete component must be a finite integer in [0, num_categories).",
        )
        locs = jnp.einsum("kij,j->ki", self.A, x_cont)
        if self.bias is not None:
            locs = locs + self.bias
        if self.B is not None and u is not None:
            locs = locs + jnp.einsum("kij,j->ki", self.B, u)
        return MixedStateDistribution(
            categorical_probs=self.transition_matrix[z],
            continuous_locs=locs,
            continuous_covs=self.cov,
            rounding=self.rounding,
        )


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
