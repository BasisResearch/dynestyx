"""Shared typing helpers for dynamical systems."""

import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from jaxtyping import Array, Int, Real


@runtime_checkable
class FunctionOfTime(Protocol):
    def __call__(
        self, t: float | int | Real[Array, ""]
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        raise NotImplementedError()


@dataclasses.dataclass
class EvaluationResult:
    """Outputs computed by an evaluation handler.

    Evaluation handlers attach this object to the ``ConditionedResult`` they
    consume.  NumPyro registration remains deferred so ``dsx.condition`` stays
    side-effect free while ``dsx.sample`` can register the same outputs later.
    """

    observation_scores: dict[str, Real[Array, "..."]] = dataclasses.field(
        default_factory=dict
    )
    _register_numpyro_sites: Callable[[str], None] | None = dataclasses.field(
        default=None, repr=False
    )


@dataclasses.dataclass
class ConditionedResult:
    """Common base for results from the NumPyro-free conditioning primitive.

    ``dsx.condition`` returns this type under both ``Filter`` and ``Smoother``.
    It carries the complete inference output through the handler stack without
    registering NumPyro sites. Filter results may additionally expose the
    canonical one-step-ahead ``predicted_observations`` and an
    ``evaluation_result`` attached by an outer evaluation handler. Posterior
    distributions are time-major:
    ``dists[i]`` corresponds to ``times[..., i]``. Leading axes on ``times``
    are optional plate axes, while each distribution may carry the matching
    plate axes in its batch shape.
    """

    marginal_loglik: Real[Array, "*plate"] | None = None
    times: Real[Array, "*time_plate time"] | None = None
    states: object = None
    dists: list | None = None
    predicted_observations: object = None
    evaluation_result: EvaluationResult | None = None
    _register_numpyro_sites: Callable[[str], None] | None = dataclasses.field(
        default=None, repr=False
    )

    def __call__(
        self, t: float | int | Real[Array, ""]
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        raise NotImplementedError(
            "ConditionedResult is not callable as a FunctionOfTime. "
            "Access .marginal_loglik, .times, .states, or .dists instead."
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

    For a discrete-time model with
    ``dynamics.observation_control_alignment="previous_transition"``, ``x_0``
    is ``None`` -- the seed state is not part of the rollout output -- and
    ``times``, ``states``, and ``observations`` are all exactly the length of
    the ``ctrl_values`` the caller supplied (one shorter than
    ``predict_times``, since :math:`y_0` is never sampled under this
    convention). ``x``, ``y``, ``u``, and ``t`` are therefore all the same
    shape, with no :math:`t_0`/:math:`x_0` remnant anywhere in the result. See
    [DiscreteTimeSimulator][dynestyx.simulation.discrete.DiscreteTimeSimulator].
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
