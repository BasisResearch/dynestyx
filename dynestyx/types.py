"""Shared typing helpers for dynamical systems."""

import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int, Real


@runtime_checkable
class FunctionOfTime(Protocol):
    def __call__(
        self, t: float | int | Real[Array, ""]
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        raise NotImplementedError()


type _Drift = Callable[
    [
        Real[Array, " state_dim"] | Real[Array, ""],
        Real[Array, " control_dim"] | Real[Array, ""] | None,
        float | int | Real[Array, ""],
    ],
    Real[Array, " state_dim"] | Real[Array, ""],
]


class ImExDrift(eqx.Module):
    """
    Split explicit/implicit drift for IMEX (implicit-explicit) solvers.

    Wraps two drift terms so a single object serves two roles:

    1. As a drop-in `drift` for `ContinuousTimeStateEvolution` used with an
       ordinary explicit solver (e.g. `diffrax.Tsit5`) or a fully-implicit
       solver (e.g. `diffrax.Kvaerno4`) -- `__call__` returns the combined
       vector field $f(x, u, t) = f_{ex}(x, u, t) + f_{im}(x, u, t)$.
    2. As the source of the explicit/implicit split consumed by diffrax's
       IMEX solvers (e.g. `diffrax.KenCarp3/4/5`, `diffrax.Sil3`, which
       require diffrax's `MultiTerm`), via `make_imex_tuple`.

    Construct with keyword-only arguments to avoid silently swapping the two
    terms: `ImExDrift(explicit_term=..., implicit_term=...)`. Positional
    construction is intentionally unsupported -- getting explicit/implicit
    backwards silently produces different (wrong) dynamics with no error,
    since both terms share the same `(x, u, t) -> state_dim` signature.

    Wherever a plain `Drift`-shaped callable `(x, u, t) -> R^{d_x}` is
    expected (see `dynestyx.models.core.Drift`), an `ImExDrift` instance may
    be used directly. When used with a solver that doesn't request diffrax's
    `MultiTerm` (i.e. not an IMEX solver),
    `dynestyx.solvers.odes.solve_ode_state_path` emits a warning naming the
    solver, since the explicit/implicit split goes unused.

    Attributes:
        explicit_term: Non-stiff component of the vector field, integrated
            explicitly by IMEX solvers.
        implicit_term: Stiff component of the vector field, integrated
            implicitly by IMEX solvers.

    Note:
        Cannot be combined with `potential` on `ContinuousTimeStateEvolution`
        -- doing so raises a `ValueError` when the `DynamicalModel` is
        constructed. Fold any potential-gradient contribution into
        `explicit_term` or `implicit_term` directly instead.
    """

    explicit_term: _Drift
    implicit_term: _Drift

    def __init__(self, *, explicit_term: _Drift, implicit_term: _Drift):
        self.explicit_term = explicit_term
        self.implicit_term = implicit_term

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        return self.explicit_term(x, u, t) + self.implicit_term(x, u, t)

    def make_imex_tuple(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> tuple[
        Real[Array, " state_dim"] | Real[Array, ""],
        Real[Array, " state_dim"] | Real[Array, ""],
    ]:
        """Return `(explicit_term(x, u, t), implicit_term(x, u, t))`.

        Order matches diffrax's `MultiTerm(ODETerm(explicit), ODETerm(implicit))`
        convention used by `solve_ode_state_path`.
        """
        return self.explicit_term(x, u, t), self.implicit_term(x, u, t)


@dataclasses.dataclass
class ConditionedResult:
    """Result of dsx.condition — the numpyro-free conditioning primitive.

    Carries all outputs from the handler stack (Filter, Smoother, etc.)
    without registering any numpyro sites.
    """

    marginal_loglik: Real[Array, "*plate"] | None = None
    states: object = None
    dists: list | None = None
    _register_numpyro_sites: Callable[[str], None] | None = dataclasses.field(
        default=None, repr=False
    )

    def __call__(
        self, t: float | int | Real[Array, ""]
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        raise NotImplementedError(
            "ConditionedResult is not callable as a FunctionOfTime. "
            "Access .marginal_loglik, .states, or .dists instead."
        )


@dataclasses.dataclass
class LatentStateResult:
    """Result of latent-state construction / scoring without NumPyro side effects.

    Let ``z = state_path_params`` denote the free variables used to
    parameterize the latent trajectory, and let

    ``x = state_path = g(z)``

    denote the full reconstructed latent state path used by the probabilistic
    model. The joint density is evaluated as ``log p(x, y)`` after
    reconstructing ``x`` from ``z``.

    In simple discrete models these may match exactly. In ODE models, or when
    exact observations leave only a subset of state coordinates free, they
    generally differ.

    When explicit missing-observation augmentation is active, the result also
    carries a second latent block:

    - ``missing_obs_values`` are the free coordinates used to fill missing
      entries of ``obs_values``,
    - ``completed_obs_values`` is the dense observation array after those
      missing entries are filled back in.

    Under plates, latent-coordinate fields are stacked when all members have
    the same shape. Ragged values are returned as a flat list in plate order,
    with the rightmost plate index varying fastest.
    """

    joint_log_prob: Real[Array, "*plate"] | None = None
    state_path_params: (
        Real[Array, "*state_path_param_shape"] | list[Real[Array, "..."]] | None
    ) = None
    state_path_param_times: (
        Real[Array, "*state_path_param_time_plate state_path_param_time"]
        | list[Real[Array, "..."]]
        | None
    ) = None
    state_path_param_coordinate_indices: (
        Int[Array, "*state_path_param_plate n_state_path_params"]
        | list[Int[Array, "..."]]
        | None
    ) = None
    state_path: Real[Array, "*state_path_shape"] | None = None
    state_path_times: Real[Array, "*state_path_time_plate state_path_time"] | None = (
        None
    )
    missing_obs_values: (
        Real[Array, "*missing_obs_shape"] | list[Real[Array, "..."]] | None
    ) = None
    missing_obs_times: (
        Real[Array, "*missing_obs_time_plate n_missing_obs"]
        | list[Real[Array, "..."]]
        | None
    ) = None
    missing_obs_coordinate_indices: (
        Int[Array, "*missing_obs_plate n_missing_obs"] | list[Int[Array, "..."]] | None
    ) = None
    completed_obs_values: Real[Array, "*completed_obs_shape"] | None = None
    state_dists: list | None = None


@dataclasses.dataclass
class SimulatedResult:
    """Result of simulation without eager NumPyro side effects.

    This result therefore stores the realized state path ``x`` and observation path ``y``
    produced on the requested simulator time grid.

    For raw forward simulation, ``times``, ``x_0``, ``states``, and
    ``observations`` are populated. The field names match their NumPyro site
    suffixes. When a simulator is layered outside a Filter or Smoother for
    posterior rollout, the same result object instead carries
    ``predicted_times``, ``predicted_states``, and
    ``predicted_observations``.
    """

    times: Real[Array, "*plate n_simulations time"] | None = None
    x_0: (
        Real[Array, "*plate n_simulations state_dim"]
        | Real[Array, "*plate n_simulations"]
        | None
    ) = None
    states: (
        Real[Array, "*plate n_simulations time state_dim"]
        | Real[Array, "*plate n_simulations time"]
        | None
    ) = None
    observations: (
        Real[Array, "*plate n_simulations time observation_dim"]
        | Real[Array, "*plate n_simulations time"]
        | None
    ) = None
    predicted_times: Real[Array, "*plate n_simulations predict_time"] | None = None
    predicted_states: (
        Real[Array, "*plate n_simulations predict_time state_dim"]
        | Real[Array, "*plate n_simulations predict_time"]
        | None
    ) = None
    predicted_observations: (
        Real[Array, "*plate n_simulations predict_time observation_dim"]
        | Real[Array, "*plate n_simulations predict_time"]
        | None
    ) = None
    _register_numpyro_sites: Callable[[str], None] | None = dataclasses.field(
        default=None, repr=False
    )


def as_scalar_time_array(
    value: float | int | Real[Array, ""], *, name: str, dtype=None
) -> Real[Array, ""]:
    """Normalize a scalar time-like value to a 0-D JAX array."""
    arr = jnp.asarray(value, dtype=dtype)
    if arr.ndim != 0 or jnp.issubdtype(arr.dtype, jnp.bool_):
        raise ValueError(
            f"{name} must be a numeric scalar (Python/NumPy real or scalar JAX array)."
        )
    return arr


def chain_numpyro_site_registrations(
    *callbacks: Callable[[str], None] | None,
) -> Callable[[str], None] | None:
    """Compose deferred NumPyro site-registration callbacks in order."""
    active_callbacks = [callback for callback in callbacks if callable(callback)]
    if not active_callbacks:
        return None

    def _register(site_name: str) -> None:
        for callback in active_callbacks:
            callback(site_name)

    return _register
