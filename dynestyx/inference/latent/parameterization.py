"""Latent path parameterizations and state-path reconstruction helpers.

This module defines the map from latent inference variables to full state
trajectories.

Throughout the latent-path code, we distinguish:

- ``z = state_path_params``: the free parameters inferred by an outer
  optimizer/MCMC routine, and
- ``x = state_path = g(z)``: the fully specified state trajectory used to
  evaluate ``log p(x, y | ...)``.

Different models and observation structures induce different choices of
``z``. For example, a discrete model may use one latent state per observation
time, whereas an exact-observation state assembly may identify ``z`` with only
the free unobserved coordinates of a completed observation array.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import jax.numpy as jnp
from jax import Array

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
    assemble_completed_observations,
    canonicalize_missing_obs_values,
    prepare_missing_observation_metadata,
    prepare_observation_views,
    resolve_missing_observation_strategy,
)
from dynestyx.solvers import solve_ode
from dynestyx.utils import _build_control_path, _raise_now_or_error_if


@dataclasses.dataclass
class AssembledStatePath:
    """Reconstructed latent state path ``x = g(z)`` for one parameterization.

    Writing ``z = state_path_params`` and ``x = state_path = g(z)``, this
    object stores both the free latent variables ``z`` and the full state path
    ``x`` used by the model density ``log p(x, y | ...)``.
    """

    state_path_params: Array
    state_path_param_times: Array
    state_path_param_coordinate_indices: Array | None
    state_path: Array
    state_path_times: Array


def canonicalize_state_path_params(
    dynamics: DynamicalModel,
    state_path_params: Array,
    *,
    n_times: int,
) -> Array:
    """Canonicalize dense ``state_path_params`` so time is the leading axis.

    For discrete or discretized models, ``state_path_params`` are represented as
    a dense array whose leading axis indexes ``state_path_param_times``. For an
    ODE latent path there is only one parameter time, so callers may pass
    either a single state-shaped value or a length-1 leading time axis.
    """
    params = jnp.asarray(state_path_params)
    event_ndim = len(dynamics.initial_condition.event_shape)

    if n_times == 1:
        if params.ndim == event_ndim:
            return jnp.expand_dims(params, axis=0)
        if params.ndim == event_ndim + 1 and params.shape[0] == 1:
            return params
        raise ValueError(
            "state_path_params is incompatible with state_path_param_times. "
            "For a single parameter time, provide either one path parameter or a "
            "length-1 leading time axis."
        )

    if params.ndim < 1 or params.shape[0] != n_times:
        raise ValueError(
            "state_path_params must have a leading time axis matching "
            "state_path_param_times for discrete / discretized models."
        )
    return params


def canonicalize_completed_observation_state_params(
    state_path_params: Array,
    *,
    n_state_path_params: int,
) -> Array:
    """Canonicalize completed-observation state params to a flat vector.

    When exact observations determine the state path, ``state_path_params`` are
    a one-dimensional vector containing only the free coordinates not fixed by
    the observed entries of ``y``. This helper enforces that convention.
    """
    try:
        return canonicalize_missing_obs_values(
            state_path_params,
            n_missing_obs=n_state_path_params,
        )
    except ValueError as exc:
        raise ValueError(
            "Completed-observation state_path_params must be a flat vector "
            "whose length matches the number of free state coordinates."
        ) from exc


def infer_state_path_param_times(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
) -> Array:
    """Infer the time index attached to ``state_path_params``.

    Current conventions are:

    - discrete / discretized models: one parameter time per observation time,
    - deterministic continuous-time models: a single parameter time at ``t0``
      because the only free latent parameter is the initial condition.
    """
    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if dynamics.t0 is None:
            raise ValueError(
                "Deterministic continuous-time latent-state assembly requires "
                "dynamics.t0 to be resolved before inferring path parameter times."
            )
        obs_times_arr = jnp.asarray(obs_times)
        return jnp.asarray([jnp.asarray(dynamics.t0, dtype=obs_times_arr.dtype)])
    return jnp.asarray(obs_times)


def default_ode_diffeqsolve_settings() -> dict[str, Any]:
    """Return default solver settings for ODE latent-path reconstruction.

    These settings are only used when reconstructing
    ``x = state_path = g(z)`` for deterministic continuous-time models, where
    ``z`` consists only of the initial condition and the rest of the path is
    generated by solving the ODE forward.
    """
    return {
        "solver": dfx.Tsit5(),
        "stepsize_controller": dfx.ConstantStepSize(),
        "adjoint": dfx.RecursiveCheckpointAdjoint(),
        "dt0": jnp.asarray(1e-3),
        "max_steps": 100_000,
    }


def assemble_completed_observation_state_path(
    *,
    state_path_params: Array,
    latent_metadata: MissingObservationMetadata,
    obs_times: Array,
    obs_values_filled: Array,
) -> AssembledStatePath:
    """Reconstruct the full state path from completed-observation latents.

    Here ``z = state_path_params`` is the vector of free unobserved
    coordinates. The reconstruction ``x = g(z)`` is performed by taking the
    observation-shaped dense array ``obs_values_filled`` and scattering those
    free coordinates back into the locations identified by
    ``latent_metadata.free_flat_indices``.
    """
    canonical_params = canonicalize_completed_observation_state_params(
        state_path_params,
        n_state_path_params=latent_metadata.free_flat_indices.shape[0],
    )
    state_path = assemble_completed_observations(
        obs_values_filled=jnp.asarray(obs_values_filled),
        missing_obs_values=canonical_params,
        missing_obs_metadata=latent_metadata,
    )
    obs_times_arr = jnp.asarray(obs_times)
    return AssembledStatePath(
        state_path_params=canonical_params,
        state_path_param_times=latent_metadata.missing_obs_times,
        state_path_param_coordinate_indices=(
            latent_metadata.missing_obs_coordinate_indices
        ),
        state_path=state_path,
        state_path_times=obs_times_arr,
    )


def assemble_state_path(
    dynamics: DynamicalModel,
    *,
    state_path_params: Array,
    state_path_param_times: Array,
    obs_times: Array | None = None,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    ode_diffeqsolve_settings: dict[str, Any] | None = None,
) -> AssembledStatePath:
    """Assemble a full state path ``x = g(z)`` from dense path parameters.

    This is the standard dense reconstruction path:

    - for discrete / discretized models, the dense latent block already is the
      full state path, so ``g`` is essentially the identity;
    - for deterministic continuous-time models, ``z`` contains only the
      initial condition and ``g`` solves the ODE to obtain the full path at the
      requested times.
    """
    state_path_param_times = jnp.asarray(state_path_param_times)
    _raise_now_or_error_if(
        state_path_param_times,
        state_path_param_times.shape[0] < 1,
        "state_path_param_times must contain at least one time point.",
    )

    canonical_params = canonicalize_state_path_params(
        dynamics,
        state_path_params,
        n_times=state_path_param_times.shape[0],
    )

    if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
        raise ValueError(
            "Latent-state assembly does not yet support native SDE models. "
            "Please discretize the model first."
        )

    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if state_path_param_times.shape[0] != 1:
            raise ValueError(
                "Deterministic continuous-time models expect exactly one latent "
                "path parameter: the initial condition."
            )

        if obs_times is None:
            state_path_times = state_path_param_times
            state_path = canonical_params
        else:
            state_path_times = jnp.concatenate(
                [state_path_param_times, jnp.asarray(obs_times)],
                axis=0,
            )
            if ctrl_times is not None and ctrl_values is not None:
                control_path = _build_control_path(
                    ctrl_times, ctrl_values, state_path_times
                )
                control_path_eval: Callable[[Array], Array | None] = lambda t: (
                    control_path.evaluate(t, left=False)
                )
            else:
                control_path_eval = lambda t: None

            state_path = solve_ode(
                dynamics,
                t0=state_path_param_times[0],
                saveat_times=state_path_times,
                x0=canonical_params[0],
                control_path_eval=control_path_eval,
                diffeqsolve_settings=(
                    ode_diffeqsolve_settings
                    if ode_diffeqsolve_settings is not None
                    else default_ode_diffeqsolve_settings()
                ),
            )

        return AssembledStatePath(
            state_path_params=canonical_params,
            state_path_param_times=state_path_param_times,
            state_path_param_coordinate_indices=None,
            state_path=state_path,
            state_path_times=state_path_times,
        )

    return AssembledStatePath(
        state_path_params=canonical_params,
        state_path_param_times=state_path_param_times,
        state_path_param_coordinate_indices=None,
        state_path=canonical_params,
        state_path_times=state_path_param_times,
    )


@dataclasses.dataclass
class StatePathParameterization:
    """Define the free state-path variables ``z = state_path_params``.

    This object only answers questions about the latent block ``z`` itself:

    - which times index ``z``,
    - whether ``z`` is dense or a flat vector of free coordinates, and
    - how to canonicalize/example that latent block.

    It does not decide how missing observations are handled or how the full
    state path ``x = g(z)`` is assembled. Those decisions live in
    :class:`ObservationCompletionPlan` and :class:`StateAssemblyPlan`.
    """

    state_path_param_times: Array
    state_path_param_coordinate_indices: Array | None = None
    n_state_path_params: int | None = None

    @property
    def uses_flat_state_path_params(self) -> bool:
        """Return whether ``state_path_params`` are a flat free-coordinate vector."""
        return self.n_state_path_params is not None

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
    """Define any separate missing-observation latent block.

    This plan is only about an optional auxiliary latent block
    ``missing_obs_values`` that fills missing coordinates of ``y`` before
    evaluating observation log-probabilities. It is intentionally independent
    from state-path assembly.
    """

    missing_obs_metadata: MissingObservationMetadata | None = None
    dense_missing_obs_shape: tuple[int, ...] | None = None

    @property
    def uses_missing_obs_augmentation(self) -> bool:
        """Return whether a separate ``missing_obs_values`` block is active."""
        return (
            self.missing_obs_metadata is not None
            or self.dense_missing_obs_shape is not None
        )

    @property
    def uses_dense_missing_obs_augmentation(self) -> bool:
        """Return whether augmentation uses an observation-shaped dense block."""
        return self.dense_missing_obs_shape is not None

    def canonicalize_missing_obs_values(self, missing_obs_values: Array) -> Array:
        """Canonicalize the auxiliary ``missing_obs_values`` latent block."""
        if self.dense_missing_obs_shape is not None:
            dense_missing_obs_values = jnp.asarray(missing_obs_values)
            if tuple(dense_missing_obs_values.shape) != self.dense_missing_obs_shape:
                raise ValueError(
                    "Dense missing_obs_values must match the observation array "
                    "shape for this latent-path layout."
                )
            return dense_missing_obs_values

        metadata = self.missing_obs_metadata
        if metadata is None:
            raise ValueError(
                "This latent-path layout does not define a separate "
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
    """Define how ``x = state_path = g(z)`` is reconstructed.

    This plan answers the model-side question:

    - does the state path come directly from ``state_path_params``, or
    - do completed exact observations determine some/all of the state path?
    """

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
        ode_diffeqsolve_settings: dict[str, Any] | None = None,
    ) -> AssembledStatePath:
        """Assemble the full latent path ``x = g(z)`` from concrete latents."""
        if self.completed_obs_state_metadata is not None:
            if obs_values_filled is None:
                raise ValueError(
                    "Completed-observation latent-state assembly requires "
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
                    "Exact-observation latent-state assembly requires pre-filled "
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
    """Concrete latent-path layout for one model/observation configuration.

    This wrapper keeps the latent inference story split into three pieces:

    - :math:`z = \\text{state_path_params}` via
      :class:`StatePathParameterization`,
    - any separate missing-observation latent block via
      :class:`ObservationCompletionPlan`,
    - the reconstruction :math:`x = g(z)` via :class:`StateAssemblyPlan`.

    Most callers should work with this object rather than the lower-level
    pieces directly.
    """

    state_path_parameterization: StatePathParameterization
    observation_completion_plan: ObservationCompletionPlan = dataclasses.field(
        default_factory=ObservationCompletionPlan
    )
    state_assembly_plan: StateAssemblyPlan = dataclasses.field(
        default_factory=StateAssemblyPlan
    )

    @property
    def state_path_param_times(self) -> Array:
        """Return the times indexing ``state_path_params``."""
        return self.state_path_parameterization.state_path_param_times

    @property
    def state_path_param_coordinate_indices(self) -> Array | None:
        """Return coordinate indices when ``state_path_params`` are flat."""
        return self.state_path_parameterization.state_path_param_coordinate_indices

    @property
    def missing_obs_metadata(self) -> MissingObservationMetadata | None:
        """Return metadata for a separate ``missing_obs_values`` block, if any."""
        return self.observation_completion_plan.missing_obs_metadata

    @property
    def observations_are_exact_constraints(self) -> bool:
        """Return whether completed observations determine the state path."""
        return self.state_assembly_plan.observations_are_exact_constraints

    def canonicalize_state_path_params(
        self,
        dynamics: DynamicalModel,
        state_path_params: Array,
    ) -> Array:
        """Canonicalize ``z = state_path_params`` for this layout."""
        return self.state_path_parameterization.canonicalize_state_path_params(
            dynamics, state_path_params
        )

    def example_state_path_params(self, dynamics: DynamicalModel) -> Array:
        """Return a shape-only example ``state_path_params`` block."""
        return self.state_path_parameterization.example_state_path_params(dynamics)

    def canonicalize_missing_obs_values(self, missing_obs_values: Array) -> Array:
        """Canonicalize the auxiliary ``missing_obs_values`` block."""
        return self.observation_completion_plan.canonicalize_missing_obs_values(
            missing_obs_values
        )

    def example_missing_obs_values(self) -> Array | None:
        """Return a shape-only example ``missing_obs_values`` block, if any."""
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
        ode_diffeqsolve_settings: dict[str, Any] | None = None,
    ) -> AssembledStatePath:
        """Assemble the full latent state path ``x = g(z)`` for this layout."""
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
    """Return zero-free-coordinate metadata for fully observed arrays."""
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
    """Prepare generic completion metadata for partially observed arrays.

    This helper is intentionally model-agnostic about how the completed
    observations will later be used. It answers only:

    - which observation entries are already fixed,
    - which ones are free,
    - and whether the missingness pattern is concrete enough to index eagerly.

    Callers may then decide whether those free coordinates become
    ``missing_obs_values`` augmentation latents or directly parameterize the
    state path.
    """
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
    """Return whether the state path should come from completed observations."""
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
    """Resolve the missingness strategy for exact-observation state assembly."""
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
    """Prepare state-path pieces for exact-observation models.

    By the time this helper runs, the caller has already decided that the
    completed observations determine the state path. The remaining question is
    only how the free state-path variables should be represented:

    - as a flat vector of free coordinates when the missingness pattern is
      concrete, or
    - as a dense state-shaped array with an exact-observation overwrite mask
      when traced execution prevents eager coordinate extraction.
    """
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
    """Prepare a separate missing-observation augmentation plan."""
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
        uses_augmentation = missing_observation_strategy == "augment" or (
            missing_observation_strategy == "auto" and marginal_mode is None
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
            "data shape for this latent-path parameterization."
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
    """Prepare the full latent-path layout for one observation configuration.

    The returned layout freezes the three structural decisions needed by
    :class:`LatentPathBuilder`:

    1. what the free state-path block ``z = state_path_params`` looks like,
    2. whether a separate ``missing_obs_values`` block is needed, and
    3. how the full state path ``x = g(z)`` is reconstructed and scored.
    """
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
    "AssembledStatePath",
    "LatentPathLayout",
    "ObservationCompletionPlan",
    "StatePathParameterization",
    "StateAssemblyPlan",
    "assemble_completed_observation_state_path",
    "assemble_state_path",
    "canonicalize_completed_observation_state_params",
    "canonicalize_state_path_params",
    "default_ode_diffeqsolve_settings",
    "infer_state_path_param_times",
    "prepare_latent_path_layout",
]
