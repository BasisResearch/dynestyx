"""Observation model implementations."""

from collections.abc import Callable
from typing import Any, NamedTuple, cast

import equinox as eqx
import jax.numpy as jnp
from jax.experimental import sparse as jax_sparse
from jaxtyping import Array, Float, Real
from numpyro import distributions as dist

from dynestyx.distributions._gaussian import covariance_matrix, normalize_covariance
from dynestyx.models.core import ObservationModel
from dynestyx.models.layout import Layout

_UNSET_COVARIANCE = object()


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
        cov: Any = _UNSET_COVARIANCE,
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
        *,
        R: Any = _UNSET_COVARIANCE,
    ):
        """
        Args:
            H (jax.Array | jax.experimental.sparse.JAXSparse | Callable): Observation
                matrix with shape $(d_y, d_x)$, or a callable `(t,)` returning it. May be
                a sparse (e.g. `BCOO`) array for `EnKFConfig`/`PFConfig`/`EKFConfig`
                (EKF works but likely gives no efficiency gain, warns) and for
                `KFConfig(filter_source="cd_dynamax")`; raises for
                `KFConfig(filter_source="cuthbert")`, which cannot support a sparse `H`.
            cov (jax.Array | Callable): Observation noise covariance with shape
                $(d_y, d_y)$, or a callable `(t,)` returning it.
            R (jax.Array | Callable): Legacy alias for cov. Providing both
                raises TypeError.
            D (jax.Array | Callable | None): Optional control matrix with
                shape $(d_y, d_u)$, or a callable `(t,)` returning it. If
                None, no control contribution is used.
            bias (jax.Array | Callable | None): Optional additive bias with
                shape $(d_y,)$, or a callable `(t,)` returning it.
        """
        if cov is not _UNSET_COVARIANCE and R is not _UNSET_COVARIANCE:
            raise TypeError("Provide only one of cov or R, not both.")
        if cov is _UNSET_COVARIANCE:
            if R is _UNSET_COVARIANCE:
                raise TypeError("LinearGaussianObservation requires cov (or legacy R).")
            cov = R
        self.H = H
        self.D = D
        self.R = cov
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
    observation noise covariance, supplied as ``cov``. The legacy keyword
    ``R`` is also accepted; supplying both raises an error.

    When an ``observation_layout`` is provided, ``cov`` must be a scalar variance
    or a matching structure of pointwise variances (independent Gaussians).
    Full covariance inputs are not supported with a layout. Accepted variances
    are expanded to dense covariance matrices internally.
    """

    h: Callable
    R: Array
    _diagonal: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        h: Callable,
        cov: Any = _UNSET_COVARIANCE,
        *,
        R: Any = _UNSET_COVARIANCE,
        state_layout: Layout | None = None,
        observation_layout: Layout | None = None,
    ):
        if cov is not _UNSET_COVARIANCE and R is not _UNSET_COVARIANCE:
            raise TypeError("Provide only one of cov or R, not both.")
        if cov is _UNSET_COVARIANCE:
            if R is _UNSET_COVARIANCE:
                raise TypeError("GaussianObservation requires cov (or legacy R).")
            cov = R
        self.h = h
        self.state_layout = state_layout
        self.observation_layout = observation_layout
        covariance, diagonal = normalize_covariance(cov, observation_layout)
        # A scalar needs the output width, which may only be known when the
        # observation is attached to a DynamicalModel.
        dimension = (
            observation_layout.state_dim
            if observation_layout is not None
            else (covariance.shape[-1] if covariance.ndim else None)
        )
        self.R = (
            covariance_matrix(covariance, diagonal, dimension)
            if dimension is not None
            else covariance
        )

    def resolve_covariance(self, observation_dim):
        """Return a copy with scalar observation variance expanded to dense R."""
        if jnp.ndim(self.R) == 0:
            return eqx.tree_at(
                lambda model: model.R,
                self,
                covariance_matrix(self.R, True, observation_dim),
            )
        return self

    def mean(self, x, u, t):
        """Return the flat conditional observation mean."""
        return self._flatten_observation(self.h(x, u, t))

    def __call__(self, x, u, t):
        loc = jnp.atleast_1d(self.mean(x, u, t))
        covariance = covariance_matrix(self.R, jnp.ndim(self.R) == 0, loc.shape[-1])
        return dist.MultivariateNormal(loc=loc, covariance_matrix=covariance)


class DiracObservation(ObservationModel):
    """Exact observations through an optionally structured observation operator."""

    h: Callable

    def __init__(
        self,
        h: Callable,
        *,
        state_layout: Layout | None = None,
        observation_layout: Layout | None = None,
    ):
        self.h = h
        self.state_layout = state_layout
        self.observation_layout = observation_layout

    def mean(self, x, u, t):
        return self._flatten_observation(self.h(x, u, t))

    def __call__(self, x, u, t):
        loc = jnp.asarray(self.mean(x, u, t))
        return dist.Delta(loc, event_dim=0 if loc.ndim == 0 else 1)


class DiracIdentityObservation(ObservationModel):
    """
    Noise-free identity observation model.

    Observations are modeled as

    $$
    y_t \\sim \\delta(x_t),
    $$


    i.e., the observation equals the latent state almost surely.
    """

    def __init__(self, *, state_layout: Layout | None = None):
        self.state_layout = state_layout
        self.observation_layout = state_layout

    def __call__(self, x, u, t):
        x = self._flatten_state(x)
        # Treat scalar latent states as scalar events, and otherwise use only
        # the trailing state axis as the event dimension so any leading batch
        # or plate axes are preserved.
        event_dim = 0 if jnp.ndim(x) == 0 else 1
        return dist.Delta(x, event_dim=event_dim)
