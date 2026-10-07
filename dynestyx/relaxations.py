"""Gaussian relaxation of selected discrete-time model components."""

import copy
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax.typing import ArrayLike
from jaxtyping import Array, Float, Real

from dynestyx.handlers import (
    HandlesSelf,
    _condition_intp,
    _dynestyx_stack_kind,
    _DynestyxStackKind,
)
from dynestyx.models.core import (
    DynamicalModel,
    ObservationModelLike,
    StateEvolutionLike,
)
from dynestyx.models.covariances import (
    Covariance,
    DiagonalCovariance,
    FullCovariance,
    ScalarCovariance,
    covariance_matrix,
)
from dynestyx.models.observations import (
    DeterministicObservation,
    GaussianObservation,
    LinearGaussianObservation,
)
from dynestyx.models.state_evolution import (
    DeterministicStateEvolution,
    GaussianStateEvolution,
    LinearGaussianStateEvolution,
)
from dynestyx.types import ConditionedResult, LatentStateResult, SimulatedResult
from dynestyx.utils.validation import _validate_array


def _parse_covariance_setting(
    value: Covariance | Real[ArrayLike, "..."] | None,
) -> Covariance | None:
    """Interpret a relaxation setting as a covariance specification.

    Scalar shorthand specifies a variance, a vector specifies diagonal variances,
    and a matrix specifies a full covariance. Batched settings use explicit
    covariance objects to distinguish leading plate axes from event axes.
    ``None`` selects no change; callable settings are rejected.
    """
    if value is None or isinstance(value, Covariance):
        return value
    if callable(value):
        raise TypeError(
            "GaussianRelaxation settings must be constant or learned arrays, not callables."
        )
    value = jnp.asarray(value)
    if value.ndim == 0:
        return ScalarCovariance(variance=value)
    if value.ndim == 1:
        return DiagonalCovariance(variance=value)
    if value.ndim == 2:
        return FullCovariance(value)
    raise ValueError(
        "Use an explicit covariance object for batched relaxation settings."
    )


class _AddedCovariance(eqx.Module):
    """Keep an existing covariance callable and added noise sliceable by plates."""

    original: Callable[..., Covariance | Real[ArrayLike, "*batch event_dim event_dim"]]
    addition: Covariance
    event_dim: int = eqx.field(static=True)

    def __call__(
        self, *args: Real[Array, "..."] | float | int | None
    ) -> Float[Array, "*batch event_dim event_dim"]:
        """Evaluate the original model covariance and add the specified noise matrix."""
        base = covariance_matrix(self.original(*args), self.event_dim)
        return _validate_array(
            base + self.addition.as_matrix(self.event_dim),
            name="Covariance",
            symmetric=True,
            atol=1e-7,
        )


def _apply_covariance_setting(
    original: Covariance
    | Real[ArrayLike, "*batch event_dim event_dim"]
    | Callable[..., Covariance | Real[ArrayLike, "*batch event_dim event_dim"]]
    | None,
    addition: Covariance,
    event_dim: int,
    mode: Literal["replace", "add"],
) -> FullCovariance | _AddedCovariance:
    """Apply a parsed covariance setting by replacing or adding to the original.

    Expand the setting to ``event_dim``. In add mode, keep existing covariance
    functions callable so the specified noise is added to each evaluated result.
    """
    matrix = addition.as_matrix(event_dim)
    if mode == "add" and original is not None:
        if callable(original):
            return _AddedCovariance(original, addition, event_dim)
        matrix = covariance_matrix(original, event_dim) + matrix
    # Preserve explicit matrix event axes when plate sizes coincide with the
    # state dimension. Backend adapters materialize the matrix when needed.
    return FullCovariance(matrix)


def _relax_initial_condition(
    initial: dist.Distribution,
    covariance: Covariance,
    state_dim: int,
    mode: Literal["replace", "add"],
) -> dist.Normal | dist.MultivariateNormal:
    """Relax a supported scalar/vector initial distribution while retaining its mean."""
    base = initial
    while isinstance(base, (dist.Independent, dist.ExpandedDistribution)):
        base = base.base_dist
    if not isinstance(base, (dist.Delta, dist.Normal, dist.MultivariateNormal)):
        raise TypeError(
            f"Cannot relax initial_condition of type {type(base).__name__}; expected Delta or a supported Gaussian."
        )
    if len(initial.event_shape) > 1:
        raise ValueError(
            "Gaussian relaxation supports scalar or vector initial events."
        )
    mean = jnp.asarray(initial.mean)
    scalar = not initial.event_shape and state_dim == 1
    original: Real[Array, "*batch state_dim state_dim"] | None = None
    if mode == "add" and not isinstance(base, dist.Delta):
        if isinstance(base, dist.MultivariateNormal):
            original = jnp.broadcast_to(
                base.covariance_matrix, mean.shape[:-1] + (state_dim, state_dim)
            )
        elif scalar:
            original = jnp.asarray(initial.variance)[..., None, None]
        else:
            original = jnp.asarray(initial.variance)[..., :, None] * jnp.eye(state_dim)
    matrix = covariance_matrix(
        _apply_covariance_setting(original, covariance, state_dim, mode),
        state_dim,
    )
    if scalar:
        return dist.Normal(mean, jnp.sqrt(matrix[..., 0, 0]))
    return dist.MultivariateNormal(mean, covariance_matrix=matrix)


def _relax_state_evolution(
    evolution: StateEvolutionLike,
    covariance: Covariance,
    state_dim: int,
    mode: Literal["replace", "add"],
) -> GaussianStateEvolution | LinearGaussianStateEvolution:
    """Relax a transition while retaining its mean function and Gaussian class."""
    if isinstance(evolution, DeterministicStateEvolution):
        return GaussianStateEvolution(
            evolution.F,
            _apply_covariance_setting(None, covariance, state_dim, mode),
        )
    if isinstance(evolution, (GaussianStateEvolution, LinearGaussianStateEvolution)):
        cov = _apply_covariance_setting(evolution.cov, covariance, state_dim, mode)
        return eqx.tree_at(lambda component: component.cov, evolution, cov)
    raise TypeError(
        f"Cannot relax state_evolution of type {type(evolution).__name__}; expected a deterministic or Gaussian state evolution."
    )


def _relax_observation(
    observation: ObservationModelLike,
    covariance: Covariance,
    observation_dim: int,
    mode: Literal["replace", "add"],
) -> GaussianObservation | LinearGaussianObservation:
    """Relax an observation while retaining its mean function and Gaussian class."""
    if isinstance(observation, DeterministicObservation):
        return GaussianObservation(
            observation.h,
            _apply_covariance_setting(None, covariance, observation_dim, mode),
        )
    if isinstance(observation, (GaussianObservation, LinearGaussianObservation)):
        cov = _apply_covariance_setting(
            observation.R, covariance, observation_dim, mode
        )
        return eqx.tree_at(lambda component: component.R, observation, cov)
    raise TypeError(
        f"Cannot relax observation_model of type {type(observation).__name__}; expected a deterministic or Gaussian observation."
    )


def relax_dynamics(
    dynamics: DynamicalModel,
    *,
    initial_condition_cov: Covariance | Real[ArrayLike, "..."] | None = None,
    state_evolution_cov: Covariance | Real[ArrayLike, "..."] | None = None,
    observation_model_cov: Covariance | Real[ArrayLike, "..."] | None = None,
    mode: Literal["replace", "add"] = "replace",
) -> DynamicalModel:
    """Return a Gaussian relaxation of selected discrete-time components.

    ``None`` leaves a component unchanged. A scalar setting is a variance;
    a vector contains diagonal variances; a matrix is a covariance.
    Use explicit :class:`ScalarCovariance`, :class:`DiagonalCovariance`, or
    :class:`FullCovariance` objects for batched settings. Settings may be learned
    JAX arrays, but may not be callables.

    ``mode='replace'`` sets the selected covariance. ``mode='add'`` adds it to
    the existing covariance (zero for deterministic components), preserving
    the state/control/time dependence of existing model covariance functions.
    Means and all model metadata are preserved;
    the input model is never mutated. Covariance objects check shape, finite
    values, and symmetry; callers supply the positive-definite effective
    covariance needed by ordinary Gaussian densities. Filter jitter settings
    retain their existing behavior.
    """
    if mode not in ("replace", "add"):
        raise ValueError("GaussianRelaxation mode must be 'replace' or 'add'.")
    if dynamics.continuous_time:
        raise TypeError(
            "GaussianRelaxation requires a discrete-time DynamicalModel; discretize continuous-time dynamics explicitly first."
        )
    ic_cov = _parse_covariance_setting(initial_condition_cov)
    state_cov = _parse_covariance_setting(state_evolution_cov)
    obs_cov = _parse_covariance_setting(observation_model_cov)
    ic, evo, obs = (
        dynamics.initial_condition,
        dynamics.state_evolution,
        dynamics.observation_model,
    )
    if ic_cov is not None:
        ic = _relax_initial_condition(ic, ic_cov, dynamics.state_dim, mode)
    if state_cov is not None:
        evo = _relax_state_evolution(evo, state_cov, dynamics.state_dim, mode)
    if obs_cov is not None:
        obs = _relax_observation(obs, obs_cov, dynamics.observation_dim, mode)
    if ic_cov is None and state_cov is None and obs_cov is None:
        return dynamics
    relaxed = copy.copy(dynamics)
    object.__setattr__(relaxed, "initial_condition", ic)
    object.__setattr__(relaxed, "state_evolution", evo)
    object.__setattr__(relaxed, "observation_model", obs)
    return relaxed


class GaussianRelaxation(ObjectInterpretation, HandlesSelf):
    """Relax selected components before inference or simulation.

    Nest this inside Filter/Smoother/Simulator and outside plates and any
    Discretizer. Arguments and covariance units match :func:`relax_dynamics`.

    Example:
        >>> with Filter(config), GaussianRelaxation(observation_model_cov=0.1):
        ...     result = condition("trajectory", dynamics, obs_times=t, obs_values=y)
    """

    def __init__(
        self,
        initial_condition_cov: Covariance | Real[ArrayLike, "..."] | None = None,
        state_evolution_cov: Covariance | Real[ArrayLike, "..."] | None = None,
        observation_model_cov: Covariance | Real[ArrayLike, "..."] | None = None,
        mode: Literal["replace", "add"] = "replace",
    ) -> None:
        """Configure the components to relax and the replacement/addition mode."""
        super().__init__()
        if mode not in ("replace", "add"):
            raise ValueError("GaussianRelaxation mode must be 'replace' or 'add'.")
        self.initial_condition_cov = _parse_covariance_setting(initial_condition_cov)
        self.state_evolution_cov = _parse_covariance_setting(state_evolution_cov)
        self.observation_model_cov = _parse_covariance_setting(observation_model_cov)
        self.mode = mode

    @implements(_dynestyx_stack_kind)
    def _stack_kind(self, **kwargs: Any) -> list[_DynestyxStackKind]:
        """Register relaxation between discretization and inference."""
        return [_DynestyxStackKind.GAUSSIAN_RELAXATION, *fwd()]

    @implements(_condition_intp)
    def _condition(
        self, name: str, dynamics: DynamicalModel, **kwargs: Any
    ) -> ConditionedResult | LatentStateResult | SimulatedResult:
        """Forward the relaxed model to the next handler with the original arguments."""
        relaxed = relax_dynamics(
            dynamics,
            initial_condition_cov=self.initial_condition_cov,
            state_evolution_cov=self.state_evolution_cov,
            observation_model_cov=self.observation_model_cov,
            mode=self.mode,
        )
        return fwd(name, relaxed, **kwargs)
