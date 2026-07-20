"""Shared typing helpers for dynamical systems."""

import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Real


@runtime_checkable
class FunctionOfTime(Protocol):
    def __call__(
        self, t: float | int | Real[Array, ""]
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        raise NotImplementedError()


@dataclasses.dataclass
class ConditionedResult:
    """Result of dsx.condition — the numpyro-free conditioning primitive.

    Carries all outputs from the handler stack (Filter, Smoother, etc.)
    without registering any numpyro sites.
    """

    marginal_loglik: jax.Array | None = None
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
    """

    joint_log_prob: jax.Array | None = None
    state_path_params: object = None
    state_path_param_times: object = None
    state_path_param_coordinate_indices: object = None
    state_path: object = None
    state_path_times: object = None
    missing_obs_values: object = None
    missing_obs_times: object = None
    missing_obs_coordinate_indices: object = None
    completed_obs_values: object = None
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

    times: Array | None = None
    x_0: Array | None = None
    states: Array | None = None
    observations: Array | None = None
    predicted_times: Array | None = None
    predicted_states: Array | None = None
    predicted_observations: Array | None = None
    _register_numpyro_sites: Callable[[str], None] | None = dataclasses.field(
        default=None, repr=False
    )


def as_scalar_time_array(
    value: float | int | Array, *, name: str, dtype=None
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
