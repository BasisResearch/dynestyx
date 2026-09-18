"""State evolution implementations.

Specialty implementations for discrete-time systems. Structure allows future
extension to LTI factories, Neural SDEs, etc.
"""

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, Real

from dynestyx.distributions._gaussian import covariance_matrix, normalize_covariance
from dynestyx.models.core import DiscreteTimeStateEvolution
from dynestyx.models.drifts import AffineDrift as _AffineDrift
from dynestyx.models.layout import Layout


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


class GaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    Nonlinear Gaussian discrete-time state transition.

    The next state is modeled as

    $$
    x_{t_{k+1}} \\sim \\mathcal{N}(F(x_{t_k}, u_{t_k}, t_k, t_{k+1}), Q),
    $$

    where $F$ is a user-provided transition function and $Q$ is the
    process-noise covariance (either constant or state/time dependent).

    When a `state_layout` is provided, the transition function `F` must accept and return structured states matching the layout and the covariance must be a scalar variance or match the structured state layout;
    full covariance matrices are not supported with a layout.
    """

    F: Callable
    cov: Any
    _diagonal: bool = eqx.field(static=True, default=False)

    def __init__(self, F: Callable, cov, *, state_layout: Layout | None = None):
        self.F = F
        self.state_layout = state_layout
        if callable(cov):
            self.cov = cov
        else:
            covariance, diagonal = normalize_covariance(cov, state_layout)
            dimension = (
                state_layout.state_dim
                if state_layout is not None
                else (covariance.shape[-1] if covariance.ndim else None)
            )
            self.cov = (
                covariance_matrix(covariance, diagonal, dimension)
                if dimension is not None
                else covariance
            )

    def resolve_covariance(self, state_dim):
        """Return a copy with scalar process variance expanded to dense covariance."""
        if not callable(self.cov) and jnp.ndim(self.cov) == 0:
            return eqx.tree_at(
                lambda model: model.cov,
                self,
                covariance_matrix(self.cov, True, state_dim),
            )
        return self

    def mean(self, x, u, t_now, t_next):
        """Return the flat conditional mean, adapting the user's state layout."""
        return self._flatten_state(self.F(x, u, t_now, t_next))

    def __call__(self, x, u, t_now, t_next):
        loc = jnp.atleast_1d(self.mean(x, u, t_now, t_next))
        covariance = self.cov
        diagonal = False if callable(covariance) else jnp.ndim(covariance) == 0
        if callable(covariance):
            covariance, diagonal = normalize_covariance(
                covariance(x, u, t_now, t_next),
                self.state_layout,
            )
        covariance = covariance_matrix(covariance, diagonal, loc.shape[-1])
        return dist.MultivariateNormal(loc=loc, covariance_matrix=covariance)


class DiracStateEvolution(DiscreteTimeStateEvolution):
    """Deterministic discrete transition."""

    F: Callable

    def __init__(self, F: Callable, *, state_layout: Layout | None = None):
        self.F = F
        self.state_layout = state_layout

    def mean(self, x, u, t_now, t_next):
        return self._flatten_state(self.F(x, u, t_now, t_next))

    def __call__(self, x, u, t_now, t_next):
        loc = jnp.asarray(self.mean(x, u, t_now, t_next))
        return dist.Delta(loc, event_dim=0 if loc.ndim == 0 else 1)
