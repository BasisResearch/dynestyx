"""Observation model implementations."""

from collections.abc import Callable
from typing import NamedTuple, cast

import jax.numpy as jnp
from jaxtyping import Array, Float, Real
from numpyro import distributions as dist

from dynestyx.models.core import ObservationModel


class LinearGaussianObservationParams(NamedTuple):
    """Linear-Gaussian observation parameters resolved at one time.

    Returned by `LinearGaussianObservation.params_at`: any callable
    (time-varying) parameter has been evaluated at the requested time, so
    every entry is a plain array (or `None` for an absent optional term).

    Expected shapes match the `LinearGaussianObservation` fields; they are
    deliberately not enforced here because plate slicing can legally hand a
    member-sliced (reduced-rank) parameter to `__call__`.
    """

    H: Float[Array, "..."]
    D: Float[Array, "..."] | None
    bias: Float[Array, "..."] | None
    R: Float[Array, "..."]


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

    Each parameter may be a constant array (time-invariant) or a callable
    `(t,) -> value` evaluated at each observation time (time-varying);
    constant and callable parameters may be mixed freely. The single time
    argument mirrors the observation model contract $p(y_t | x_t, u_t, t)$
    (transitions span an interval; observations happen at one time).

    Note:
        - Callable parameters receive only the observation time `t`; they
          must not depend on state or controls (use `GaussianObservation`
          for nonlinear measurement functions).
        - Callables must be pure, JAX-traceable functions returning a fixed
          shape.
        - Backend support: time-varying parameters work with the simulators
          and the `filter_source="cuthbert"` filters/smoothers; the
          cd_dynamax backend requires constant arrays and raises `TypeError`
          otherwise.
    """

    H: (
        Float[Array, "*h_plate observation_dim state_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*h_plate observation_dim state_dim"],
        ]
    )
    R: (
        Float[Array, "*r_plate observation_dim observation_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*r_plate observation_dim observation_dim"],
        ]
    )
    D: (
        Float[Array, "*d_matrix_plate observation_dim control_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*d_matrix_plate observation_dim control_dim"],
        ]
        | None
    ) = None
    bias: (
        Float[Array, "*bias_plate observation_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*bias_plate observation_dim"],
        ]
        | None
    ) = None

    def __init__(
        self,
        H: Float[Array, "*h_plate observation_dim state_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*h_plate observation_dim state_dim"],
        ],
        R: Float[Array, "*r_plate observation_dim observation_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*r_plate observation_dim observation_dim"],
        ],
        D: Float[Array, "*d_matrix_plate observation_dim control_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*d_matrix_plate observation_dim control_dim"],
        ]
        | None = None,
        bias: Float[Array, "*bias_plate observation_dim"]
        | Callable[
            [float | int | Real[Array, ""]],
            Float[Array, "*bias_plate observation_dim"],
        ]
        | None = None,
    ):
        """
        Args:
            H (jax.Array | Callable): Observation matrix with shape
                $(d_y, d_x)$, or a callable `(t,)` returning it.
            R (jax.Array | Callable): Observation noise covariance with shape
                $(d_y, d_y)$, or a callable `(t,)` returning it.
            D (jax.Array | Callable | None): Optional control matrix with
                shape $(d_y, d_u)$, or a callable `(t,)` returning it. If
                None, no control contribution is used.
            bias (jax.Array | Callable | None): Optional additive bias with
                shape $(d_y,)$, or a callable `(t,)` returning it.
        """
        self.H = H
        self.D = D
        self.R = R
        self.bias = bias

    @property
    def is_time_invariant(self) -> bool:
        """True iff every parameter is a constant array (no callables)."""
        return not any(callable(field) for field in (self.H, self.D, self.bias, self.R))

    def params_at(
        self, t: float | int | Real[Array, ""]
    ) -> LinearGaussianObservationParams:
        """Resolve `(H, D, bias, R)` at one observation time.

        Constant parameters are returned unchanged; callable parameters are
        evaluated at `t`.
        """

        def _resolve(field):
            if field is None or not callable(field):
                return field
            fn = cast(
                Callable[[float | int | Real[Array, ""]], Array],
                field,
            )
            return jnp.asarray(fn(t))

        return LinearGaussianObservationParams(
            H=_resolve(self.H),
            D=_resolve(self.D),
            bias=_resolve(self.bias),
            R=_resolve(self.R),
        )

    def __call__(self, x, u, t):
        H, D, bias, R = self.params_at(t)
        loc = jnp.dot(H, x)
        if D is not None and u is not None:
            loc = loc + jnp.dot(D, u)
        if bias is not None:
            loc = loc + bias
        return dist.MultivariateNormal(loc=loc, covariance_matrix=R)


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

    h: Callable[
        [
            Real[Array, " state_dim"] | Real[Array, ""],
            Real[Array, " control_dim"] | Real[Array, ""] | None,
            Real[Array, ""],
        ],
        Real[Array, " observation_dim"] | Real[Array, ""],
    ]
    R: Float[Array, "*plate observation_dim observation_dim"]

    def __init__(
        self,
        h: Callable[
            [
                Real[Array, " state_dim"] | Real[Array, ""],
                Real[Array, " control_dim"] | Real[Array, ""] | None,
                Real[Array, ""],
            ],
            Real[Array, " observation_dim"] | Real[Array, ""],
        ],
        R: Float[Array, "*plate observation_dim observation_dim"],
    ):
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
