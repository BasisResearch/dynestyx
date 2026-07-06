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

    In simple discrete models these may match exactly. In ODE or compressed
    exact-observation settings they generally differ.
    """

    joint_log_prob: jax.Array | None = None
    state_path_params: object = None
    state_path_param_times: object = None
    state_path_param_coordinate_indices: object = None
    state_path: object = None
    state_path_times: object = None
    state_dists: list | None = None
    _register_numpyro_sites: Callable[[str], None] | None = dataclasses.field(
        default=None, repr=False
    )


@dataclasses.dataclass
class SimulatedResult:
    """Result of pure-JAX forward simulation without NumPyro side effects.

    The simulator now conceptually owns data generation only. This result
    therefore stores the realized state path ``x`` and observation path ``y``
    produced on the requested simulator time grid.
    """

    times: object = None
    states: object = None
    observations: object = None
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
