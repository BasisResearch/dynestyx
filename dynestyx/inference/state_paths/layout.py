"""Pure-JAX state-path layout helpers.

This module defines the structural choices behind latent state-path inference:

- the free parameter block ``z = state_path_params``,
- any separate missing-observation completion block,
- and the reconstruction plan ``x = g(z)``.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jax import Array

from dynestyx.inference.state_paths.reconstruct import (
    AssembledStatePath,
    assemble_completed_observation_state_path,
    assemble_state_path,
    canonicalize_completed_observation_state_params,
    canonicalize_state_path_params,
    infer_state_path_param_times,
)
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
    _canonicalize_observation_distribution,
    _marginalizable_distribution_mode,
    _probe_observation_distribution,
    _supports_missing_observation_augmentation,
    canonicalize_missing_obs_values,
    prepare_missing_observation_metadata,
    prepare_observation_views,
    resolve_missing_observation_strategy,
)


@dataclasses.dataclass
class StatePathParameterization:
    """Define the free state-path variables ``z = state_path_params``."""

    state_path_param_times: Array
    state_path_param_coordinate_indices: Array | None = None
    n_state_path_params: int | None = None

    def canonicalize_state_path_params(
        self,
        dynamics: DynamicalModel,
        state_path_params: Array,
    ) -> Array:
        """Canonicalize ``z = state_path_params`` for this parameterization."""
        if self.n_state_path_params is not None:
            return canonicalize_completed_observation_state_params(
                state_path_params,
                n_state_path_params=self.n_state_path_params,
            )
        return canonicalize_state_path_params(
            dynamics,
            state_path_params,
            n_times=self.state_path_param_times.shape[0],
        )

    def example_state_path_params(self, dynamics: DynamicalModel) -> Array:
        """Return a shape-only example latent block for ``state_path_params``."""
        if self.n_state_path_params is not None:
            return jnp.zeros((self.n_state_path_params,))
        return canonicalize_state_path_params(
            dynamics,
            jnp.zeros(
                (
                    self.state_path_param_times.shape[0],
                    *dynamics.initial_condition.event_shape,
                )
            ),
            n_times=self.state_path_param_times.shape[0],
        )


@dataclasses.dataclass
class ObservationCompletionPlan:
    """Define any separate missing-observation latent block."""

    missing_obs_metadata: MissingObservationMetadata | None = None
    dense_missing_obs_shape: tuple[int, ...] | None = None

    def canonicalize_missing_obs_values(self, missing_obs_values: Array) -> Array:
        """Canonicalize the auxiliary ``missing_obs_values`` latent block."""
        if self.dense_missing_obs_shape is not None:
            dense_missing_obs_values = jnp.asarray(missing_obs_values)
            if tuple(dense_missing_obs_values.shape) != self.dense_missing_obs_shape:
                raise ValueError(
                    "Dense missing_obs_values must match the observation array "
                    "shape for this state-path layout."
                )
            return dense_missing_obs_values

        metadata = self.missing_obs_metadata
        if metadata is None:
            raise ValueError(
                "This state-path layout does not define a separate "
                "missing_obs_values latent block."
            )

        return canonicalize_missing_obs_values(
            missing_obs_values,
            n_missing_obs=metadata.free_flat_indices.shape[0],
        )

    def example_missing_obs_values(self) -> Array | None:
        """Return a shape-only example ``missing_obs_values`` block when needed."""
        if self.dense_missing_obs_shape is not None:
            return jnp.zeros(self.dense_missing_obs_shape)
        if self.missing_obs_metadata is None:
            return None
        return jnp.zeros((self.missing_obs_metadata.free_flat_indices.shape[0],))


@dataclasses.dataclass
class StateAssemblyPlan:
    """Define how ``x = state_path = g(z)`` is reconstructed."""

    completed_obs_state_metadata: MissingObservationMetadata | None = None
    completed_obs_exact_mask: Array | None = None

    @property
    def observations_are_exact_constraints(self) -> bool:
        """Return whether completed observations determine the state path."""
        return (
            self.completed_obs_state_metadata is not None
            or self.completed_obs_exact_mask is not None
        )

    def assemble_from_params(
        self,
        dynamics: DynamicalModel,
        *,
        parameterization: StatePathParameterization,
        state_path_params: Array,
        obs_times: Array,
        obs_values_filled: Array | None,
        ctrl_times: Array | None = None,
        ctrl_values: Array | None = None,
        ode_diffeqsolve_settings: dict | None = None,
    ) -> AssembledStatePath:
        """Assemble the full state path ``x = g(z)`` from concrete latents."""
        if self.completed_obs_state_metadata is not None:
            if obs_values_filled is None:
                raise ValueError(
                    "Completed-observation state-path assembly requires "
                    "pre-filled observation values."
                )
            return assemble_completed_observation_state_path(
                state_path_params=state_path_params,
                latent_metadata=self.completed_obs_state_metadata,
                obs_times=obs_times,
                obs_values_filled=obs_values_filled,
            )

        if self.completed_obs_exact_mask is not None:
            if obs_values_filled is None:
                raise ValueError(
                    "Exact-observation state-path assembly requires pre-filled "
                    "observation values."
                )
            if jnp.asarray(state_path_params).size == 0:
                return AssembledStatePath(
                    state_path_params=jnp.asarray(state_path_params),
                    state_path_param_times=parameterization.state_path_param_times,
                    state_path_param_coordinate_indices=None,
                    state_path=jnp.asarray(obs_values_filled),
                    state_path_times=jnp.asarray(obs_times),
                )
            canonical_params = parameterization.canonicalize_state_path_params(
                dynamics, state_path_params
            )
            state_path = jnp.where(
                self.completed_obs_exact_mask,
                jnp.asarray(obs_values_filled),
                canonical_params,
            )
            return AssembledStatePath(
                state_path_params=canonical_params,
                state_path_param_times=parameterization.state_path_param_times,
                state_path_param_coordinate_indices=None,
                state_path=state_path,
                state_path_times=jnp.asarray(obs_times),
            )

        return assemble_state_path(
            dynamics,
            state_path_params=state_path_params,
            state_path_param_times=parameterization.state_path_param_times,
            obs_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            ode_diffeqsolve_settings=ode_diffeqsolve_settings,
        )


@dataclasses.dataclass
class LatentPathLayout:
    """Concrete state-path layout for one model/observation configuration."""

    state_path_parameterization: StatePathParameterization
    observation_completion_plan: ObservationCompletionPlan = dataclasses.field(
        default_factory=ObservationCompletionPlan
    )
    state_assembly_plan: StateAssemblyPlan = dataclasses.field(
        default_factory=StateAssemblyPlan
    )

    @property
    def state_path_param_times(self) -> Array:
        return self.state_path_parameterization.state_path_param_times

    @property
    def state_path_param_coordinate_indices(self) -> Array | None:
        return self.state_path_parameterization.state_path_param_coordinate_indices

    @property
    def missing_obs_metadata(self) -> MissingObservationMetadata | None:
        return self.observation_completion_plan.missing_obs_metadata

    @property
    def observations_are_exact_constraints(self) -> bool:
        return self.state_assembly_plan.observations_are_exact_constraints

    def canonicalize_state_path_params(
        self,
        dynamics: DynamicalModel,
        state_path_params: Array,
    ) -> Array:
        return self.state_path_parameterization.canonicalize_state_path_params(
            dynamics, state_path_params
        )

    def example_state_path_params(self, dynamics: DynamicalModel) -> Array:
        return self.state_path_parameterization.example_state_path_params(dynamics)

    def canonicalize_missing_obs_values(self, missing_obs_values: Array) -> Array:
        return self.observation_completion_plan.canonicalize_missing_obs_values(
            missing_obs_values
        )

    def example_missing_obs_values(self) -> Array | None:
        return self.observation_completion_plan.example_missing_obs_values()

    def assemble_from_params(
        self,
        dynamics: DynamicalModel,
        *,
        state_path_params: Array,
        obs_times: Array,
        obs_values_filled: Array | None,
        ctrl_times: Array | None = None,
        ctrl_values: Array | None = None,
        ode_diffeqsolve_settings: dict | None = None,
    ) -> AssembledStatePath:
        return self.state_assembly_plan.assemble_from_params(
            dynamics,
            parameterization=self.state_path_parameterization,
            state_path_params=state_path_params,
            obs_times=obs_times,
            obs_values_filled=obs_values_filled,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            ode_diffeqsolve_settings=ode_diffeqsolve_settings,
        )


def _empty_missing_observation_metadata(
    *,
    obs_times: Array,
    obs_values: Array,
) -> MissingObservationMetadata:
    obs_times_arr = jnp.asarray(obs_times)
    obs_values_arr = jnp.asarray(obs_values)
    return MissingObservationMetadata(
        missing_obs_times=obs_times_arr[:0],
        missing_obs_coordinate_indices=(
            None if obs_values_arr.ndim == 1 else jnp.zeros((0,), dtype=jnp.int32)
        ),
        free_flat_indices=jnp.zeros((0,), dtype=jnp.int32),
        observation_shape=tuple(obs_values_arr.shape),
        has_missing=False,
        has_partial_missing=False,
        has_fully_missing_rows=False,
    )


def _prepare_completion_metadata(
    *,
    dynamics: DynamicalModel,
    obs_times: Array,
    obs_values: Array,
    obs_mask: Array | None,
    obs_has_missing: bool | None,
) -> tuple[MissingObservationMetadata | None, Array | None]:
    obs_times_arr = jnp.asarray(obs_times)
    obs_values_arr = jnp.asarray(obs_values)

    if obs_has_missing is False:
        return _empty_missing_observation_metadata(
            obs_times=obs_times_arr,
            obs_values=obs_values_arr,
        ), None

    try:
        return prepare_missing_observation_metadata(
            dynamics,
            obs_times=obs_times_arr,
            obs_values=obs_values_arr,
        ), None
    except ValueError as exc:
        if obs_mask is None:
            raise
        if "concrete missingness pattern" not in str(exc):
            raise
        return None, obs_mask


def _uses_completed_observation_state_assembly(
    dynamics: DynamicalModel,
    *,
    obs_values_filled: Array | None,
    obs_mask: Array | None,
    obs_has_missing: bool | None,
) -> bool:
    return (
        isinstance(dynamics.observation_model, DiracIdentityObservation)
        and not isinstance(
            dynamics.state_evolution, DeterministicContinuousTimeStateEvolution
        )
        and not isinstance(
            dynamics.state_evolution, StochasticContinuousTimeStateEvolution
        )
        and obs_values_filled is not None
        and (obs_has_missing is False or obs_mask is not None)
    )


def _resolve_exact_observation_strategy(
    *,
    requested_strategy: MissingObservationStrategy,
    obs_has_missing: bool | None,
) -> MissingObservationStrategy:
    if obs_has_missing is False:
        return requested_strategy
    if requested_strategy in ("marginalize", "error"):
        raise ValueError(
            "DiracIdentityObservation missingness in latent-path inference "
            "supports only augment semantics. Use "
            "missing_observation_strategy='auto' or 'augment'."
        )
    if requested_strategy == "auto":
        return "augment"
    return requested_strategy


def _prepare_completed_observation_state_layout(
    *,
    dynamics: DynamicalModel,
    obs_times: Array,
    obs_values: Array,
    obs_mask: Array | None,
    obs_has_missing: bool | None,
) -> tuple[StatePathParameterization, StateAssemblyPlan]:
    metadata, exact_mask = _prepare_completion_metadata(
        dynamics=dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_has_missing=obs_has_missing,
    )
    if metadata is not None:
        return (
            StatePathParameterization(
                state_path_param_times=metadata.missing_obs_times,
                state_path_param_coordinate_indices=(
                    metadata.missing_obs_coordinate_indices
                ),
                n_state_path_params=metadata.free_flat_indices.shape[0],
            ),
            StateAssemblyPlan(completed_obs_state_metadata=metadata),
        )

    if exact_mask is None:
        raise ValueError(
            "Exact-observation state assembly requires either concrete free "
            "coordinate metadata or an exact-observation mask."
        )

    return (
        StatePathParameterization(state_path_param_times=jnp.asarray(obs_times)),
        StateAssemblyPlan(completed_obs_exact_mask=exact_mask),
    )


def _prepare_observation_completion_plan(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_values: Array,
    obs_mask: Array | None,
    missing_observation_strategy: MissingObservationStrategy,
) -> ObservationCompletionPlan:
    plan = ObservationCompletionPlan()
    if missing_observation_strategy not in ("augment", "auto") or obs_mask is None:
        return plan

    obs_times_arr = jnp.asarray(obs_times)
    obs_values_arr = jnp.asarray(obs_values)
    observation_dim = 1 if obs_values_arr.ndim == 1 else obs_values_arr.shape[-1]
    probed_obs_dist = _canonicalize_observation_distribution(
        _probe_observation_distribution(dynamics),
        observation_dim=observation_dim,
    )
    marginal_mode = _marginalizable_distribution_mode(probed_obs_dist)
    augmentation_supported = _supports_missing_observation_augmentation(probed_obs_dist)

    if missing_observation_strategy == "auto" and marginal_mode is not None:
        return plan

    try:
        metadata, _ = _prepare_completion_metadata(
            dynamics=dynamics,
            obs_times=obs_times_arr,
            obs_values=obs_values_arr,
            obs_mask=obs_mask,
            obs_has_missing=None,
        )
    except ValueError as exc:
        if "concrete missingness pattern" not in str(exc):
            raise
        if missing_observation_strategy == "augment":
            if not augmentation_supported:
                raise NotImplementedError(
                    "Explicit missing-observation augmentation currently "
                    "requires a continuous observation family."
                ) from exc
            return ObservationCompletionPlan(
                dense_missing_obs_shape=tuple(obs_values_arr.shape)
            )
        return ObservationCompletionPlan()

    if metadata is None:
        concrete_mask_metadata = None
        try:
            concrete_mask_metadata = prepare_missing_observation_metadata(
                dynamics,
                obs_times=obs_times_arr,
                obs_mask=obs_mask,
            )
        except ValueError:
            pass

        has_partial_missing = (
            False
            if concrete_mask_metadata is None
            else concrete_mask_metadata.has_partial_missing
        )
        uses_augmentation = missing_observation_strategy == "augment" or (
            missing_observation_strategy == "auto"
            and marginal_mode is None
            and has_partial_missing
        )
        if not uses_augmentation:
            return ObservationCompletionPlan()
        if not augmentation_supported:
            raise NotImplementedError(
                "Explicit missing-observation augmentation currently "
                "requires a continuous observation family."
            )
        return ObservationCompletionPlan(
            dense_missing_obs_shape=tuple(obs_values_arr.shape)
        )

    if metadata.observation_shape != tuple(obs_values_arr.shape):
        raise ValueError(
            "Prepared missing observation metadata does not match the observed "
            "data shape for this state-path layout."
        )

    uses_augmentation, _ = resolve_missing_observation_strategy(
        dynamics,
        observation_dim=observation_dim,
        has_missing=metadata.has_missing,
        has_partial_missing=metadata.has_partial_missing,
        requested_strategy=missing_observation_strategy,
    )
    if uses_augmentation:
        return ObservationCompletionPlan(missing_obs_metadata=metadata)
    return ObservationCompletionPlan()


def prepare_latent_path_layout(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_values: Array,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    obs_values_filled: Array | None = None,
    obs_mask: Array | None = None,
    obs_has_missing: bool | None = None,
) -> LatentPathLayout:
    """Prepare the structural state-path layout for one observation configuration."""
    obs_times_arr = jnp.asarray(obs_times)
    obs_values_arr = jnp.asarray(obs_values)

    if obs_values_filled is None or obs_mask is None or obs_has_missing is None:
        obs_values_filled, obs_mask, obs_has_missing = prepare_observation_views(
            dynamics, obs_values_arr
        )

    state_path_from_completed_observations = _uses_completed_observation_state_assembly(
        dynamics,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        obs_has_missing=obs_has_missing,
    )

    if state_path_from_completed_observations:
        missing_observation_strategy = _resolve_exact_observation_strategy(
            requested_strategy=missing_observation_strategy,
            obs_has_missing=obs_has_missing,
        )
        (
            state_path_parameterization,
            state_assembly_plan,
        ) = _prepare_completed_observation_state_layout(
            dynamics=dynamics,
            obs_times=obs_times_arr,
            obs_values=obs_values_arr,
            obs_mask=obs_mask,
            obs_has_missing=obs_has_missing,
        )
        observation_completion_plan = ObservationCompletionPlan()
    else:
        state_path_parameterization = StatePathParameterization(
            state_path_param_times=infer_state_path_param_times(
                dynamics, obs_times=obs_times_arr
            )
        )
        state_assembly_plan = StateAssemblyPlan()
        observation_completion_plan = _prepare_observation_completion_plan(
            dynamics,
            obs_times=obs_times_arr,
            obs_values=obs_values_arr,
            obs_mask=obs_mask,
            missing_observation_strategy=missing_observation_strategy,
        )

    return LatentPathLayout(
        state_path_parameterization=state_path_parameterization,
        observation_completion_plan=observation_completion_plan,
        state_assembly_plan=state_assembly_plan,
    )


__all__ = [
    "LatentPathLayout",
    "ObservationCompletionPlan",
    "StateAssemblyPlan",
    "StatePathParameterization",
    "prepare_latent_path_layout",
]
