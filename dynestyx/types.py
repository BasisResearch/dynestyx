"""Shared typing helpers for dynamical systems."""

import dataclasses
from collections.abc import Callable
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Int, PyTree, Real

from dynestyx.models.layout import Layout


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

    state_layout: Layout | None = dataclasses.field(default=None, kw_only=True)
    observation_layout: Layout | None = dataclasses.field(default=None, kw_only=True)

    @property
    def structured_means(self):
        """Posterior means as a pytree; posterior distributions remain flat."""
        if not self.dists:
            return None
        means = jnp.stack(
            [
                jnp.asarray(d.mean)[..., None] if not d.event_shape else d.mean
                for d in self.dists
            ],
            axis=-2,
        )
        return (
            means if self.state_layout is None else self.state_layout.unflatten(means)
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

    state_layout: Layout | None = dataclasses.field(default=None, kw_only=True)
    observation_layout: Layout | None = dataclasses.field(default=None, kw_only=True)

    @property
    def structured_state_path(self):
        """The reconstructed latent trajectory in its declared structure."""
        if self.state_path is None or self.state_layout is None:
            return self.state_path
        return self.state_layout.unflatten(self.state_path)


class SimulatedResult(eqx.Module):
    """Simulation output in the user's declared state and observation structures.

    With layouts, ``x_0``, ``states``, ``observations``, and corresponding
    ``predicted_*`` fields contain pytrees with leading plate, simulation, and
    time axes. Without layouts they retain their existing array representation.
    ``flatten()`` returns vector arrays in the same fields for inference inputs
    and NumPyro sites; no duplicate flat fields are stored. Simulation backends construct flat results, then call ``unflatten()``.
    """

    times: Real[Array, "*plate n_simulations time"] | None = None
    x_0: PyTree[Array] | None = None
    states: PyTree[Array] | None = None
    observations: PyTree[Array] | None = None
    predicted_times: Real[Array, "*plate n_simulations predict_time"] | None = None
    predicted_states: PyTree[Array] | None = None
    predicted_observations: PyTree[Array] | None = None
    _register_numpyro_sites: Callable[[str], None] | None = eqx.field(
        default=None, repr=False, static=True
    )
    state_layout: Layout | None = eqx.field(default=None, static=True, kw_only=True)
    observation_layout: Layout | None = eqx.field(
        default=None, static=True, kw_only=True
    )

    _is_flat: bool = eqx.field(default=True, static=True, kw_only=True, repr=False)

    def unflatten(self):
        """Return a structured copy, preserving leading axes and layout metadata.

        Newly constructed results hold flat arrays. Missing fields and absent
        layouts are unchanged; calling this again on a structured result is a no-op.
        """
        return self._convert(flat=False)

    def flatten(self):
        """Return a flat copy; calling this on a flat result is a no-op."""
        return self._convert(flat=True)

    def _convert(self, *, flat):
        if self._is_flat == flat or (
            self.state_layout is None and self.observation_layout is None
        ):
            return self
        updates = {}
        for name, layout in (
            ("x_0", self.state_layout),
            ("states", self.state_layout),
            ("predicted_states", self.state_layout),
            ("observations", self.observation_layout),
            ("predicted_observations", self.observation_layout),
        ):
            value = getattr(self, name)
            if layout is not None and value is not None:
                updates[name] = (
                    layout.flatten(value) if flat else layout.unflatten(value)
                )
        return dataclasses.replace(self, **updates, _is_flat=flat)

    # Keep the explicit names from the initial structured-state implementation.
    @property
    def structured_states(self):
        return self.unflatten().states

    @property
    def structured_x_0(self):
        return self.unflatten().x_0

    @property
    def structured_observations(self):
        return self.unflatten().observations

    @property
    def structured_predicted_states(self):
        return self.unflatten().predicted_states

    @property
    def structured_predicted_observations(self):
        return self.unflatten().predicted_observations


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
