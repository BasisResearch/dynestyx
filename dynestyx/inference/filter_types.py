"""Public callback and output types for custom filtering algorithms."""

import dataclasses
from typing import NamedTuple, Protocol, runtime_checkable

import jax
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Real, Shaped


class FilterInputs(NamedTuple):
    """One observation or a sequence with a leading time axis."""

    y: Shaped[Array, "*batch_time observation_dim"]
    u: Real[Array, "*batch_time control_dim"]
    u_prev: Real[Array, "*batch_time control_dim"]
    time: Real[Array, "*batch_time"]
    time_prev: Real[Array, "*batch_time"]
    is_first_step: Bool[Array, "*batch_time"]


@runtime_checkable
class Initialize[Prior](Protocol):
    def __call__(
        self, inputs: FilterInputs, /, *, key: PRNGKeyArray | None
    ) -> Prior: ...


@runtime_checkable
class Predict[Prior, State](Protocol):
    def __call__(
        self, filtered: State, inputs: FilterInputs, /, *, key: PRNGKeyArray | None
    ) -> Prior: ...


@runtime_checkable
class Update[Prior, State](Protocol):
    def __call__(
        self, prior: Prior, inputs: FilterInputs, /, *, key: PRNGKeyArray | None
    ) -> State: ...


class SplitFilter[Prior, State](NamedTuple):
    """initialize and predict return a prior; update returns the filtered state."""

    initialize: Initialize[Prior]
    predict: Predict[Prior, State]
    update: Update[Prior, State]


@runtime_checkable
class SplitSteps[Prior, State](Protocol):
    """Structural contract for SplitFilter and algorithms with matching methods."""

    def initialize(
        self, inputs: FilterInputs, /, *, key: PRNGKeyArray | None
    ) -> Prior: ...
    def predict(
        self, filtered: State, inputs: FilterInputs, /, *, key: PRNGKeyArray | None
    ) -> Prior: ...
    def update(
        self, prior: Prior, inputs: FilterInputs, /, *, key: PRNGKeyArray | None
    ) -> State: ...


@runtime_checkable
class CombinedSteps[Prepared, State](Protocol):
    """Sequential preparation may return a different pytree from the filtered state."""

    def init_prepare(self, *, key: PRNGKeyArray | None) -> State: ...
    def filter_prepare(
        self, inputs: FilterInputs, /, *, key: PRNGKeyArray | None
    ) -> Prepared: ...
    def filter_combine(self, previous: State, prepared: Prepared, /) -> State: ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class PredictedObservationOutputs:
    """Predicted observations for one step or a plate- and time-indexed history."""

    mean: Float[Array, "*batch_time observation_dim"] | None = None
    cov: Float[Array, "*batch_time observation_dim observation_dim"] | None = None
    obs_cov: Float[Array, "*batch_time observation_dim observation_dim"] | None = None
    ensemble: Float[Array, "*batch_time n_members observation_dim"] | None = None
    obs_ensemble: Float[Array, "*batch_time n_members observation_dim"] | None = None
    noise_cov: Float[Array, "*batch_time observation_dim observation_dim"] | None = None


class FilterOutputs(NamedTuple):
    """One step or a stacked history. Supply Gaussian moments or weighted particles."""

    log_normalizing_constant: Real[Array, "*batch_time"]
    filtered_means: Real[Array, "*batch_time state_dim"] | None = None
    filtered_covariances: Real[Array, "*batch_time state_dim state_dim"] | None = None
    particles: (
        Real[Array, "*batch_time n_particles state_dim"]
        | Real[Array, "*batch_time n_particles"]
        | None
    ) = None
    log_weights: Real[Array, "*batch_time n_particles"] | None = None
    predicted_observations: PredictedObservationOutputs | None = None
    extra_outputs: dict[str, Array] | None = None
