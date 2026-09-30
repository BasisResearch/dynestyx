"""Observation model implementations."""

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse
from jaxtyping import Array, Float, Real
from numpyro import distributions as dist

from dynestyx.models.core import ObservationModel
from dynestyx.models.layout import Layouts
from dynestyx.models.utils.distribution_utils import _dirac


class LinearGaussianObservationParams(NamedTuple):
    """Linear-Gaussian observation parameters resolved at one time.

    Returned by `LinearGaussianObservation.params_at`: any callable
    (time-varying) parameter has been evaluated at the requested time, so
    every entry is a plain array (or `None` for an absent optional term).

    Expected shapes match the `LinearGaussianObservation` fields; they are
    deliberately not enforced here because plate slicing can legally hand a
    member-sliced (reduced-rank) parameter to `__call__`.
    """

    H: Float[Array, "..."] | jax_sparse.JAXSparse
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
        | jax_sparse.JAXSparse
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
        | jax_sparse.JAXSparse
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
            H (jax.Array | jax.experimental.sparse.JAXSparse | Callable): Observation
                matrix with shape $(d_y, d_x)$, or a callable `(t,)` returning it. May be
                a sparse (e.g. `BCOO`) array for `EnKFConfig`/`PFConfig`/`EKFConfig`
                (EKF works but likely gives no efficiency gain, warns) and for
                `KFConfig(filter_source="cd_dynamax")`; raises for
                `KFConfig(filter_source="cuthbert")`, which cannot support a sparse `H`.
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
        loc = H @ x
        if D is not None and u is not None:
            loc = loc + D @ u
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


class DiracObservation(ObservationModel):
    """Deterministic observation with optional structured inputs and output."""

    h: Callable[[Any, Any, Any], Any]
    layout: Layouts | None
    event_dim: int | None

    def __init__(
        self,
        h: Callable[[Any, Any, Any], Any],
        *,
        layout: Layouts | None = None,
        event_dim: int | None = None,
    ):
        self.h = h
        self.layout = layout
        self.event_dim = event_dim

    def mean(self, x, u, t):
        """Compute the observation location in flat coordinates."""
        state_layout = None if self.layout is None else self.layout.state
        control_layout = None if self.layout is None else self.layout.control
        observation_layout = None if self.layout is None else self.layout.observation
        if state_layout is not None:
            x = state_layout.unflatten(x)
        if control_layout is not None and u is not None:
            u = control_layout.unflatten(u)
        observation = self.h(x, u, t)
        return (
            jnp.asarray(observation)
            if observation_layout is None
            else observation_layout.flatten(observation)
        )

    def __call__(self, x, u, t):
        return _dirac(self.mean(x, u, t), self.event_dim)


def _identity_observation(x, u, t):
    return x


class DiracIdentityObservation(DiracObservation):
    """
    Noise-free identity observation model.

    Observations are modeled as

    $$
    y_t \\sim \\delta(x_t),
    $$

    i.e., the observation equals the latent state almost surely.
    """

    def __init__(self):
        super().__init__(_identity_observation)
