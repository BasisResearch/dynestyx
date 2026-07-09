"""Latent-path handler for joint state + parameter inference.

This module is the orchestration layer for latent-path inference. It does not
define the latent parameterization ``z`` or the scoring rules itself; instead it
coordinates three steps:

1. prepare a concrete :class:`LatentPathLayout`,
2. reconstruct and score ``x = g(z)`` in pure JAX, and
3. optionally register NumPyro sites after that pure-JAX work is done.

Keeping those responsibilities separate is what allows
``LatentPathBuilder`` to keep its reconstruction/scoring logic in pure JAX
even though the public handler itself is NumPyro-facing and only supports
``dsx.sample(...)``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.checkers import _validate_inference_supported_model_classes
from dynestyx.inference.latent.plate import (
    _plate_member_specs,
    _stack_member_attr,
)
from dynestyx.inference.latent.prepare import (
    _prepare_latent_path_request,
    _PreparedLatentPathRequest,
)
from dynestyx.inference.state_paths.reconstruct import AssembledStatePath
from dynestyx.inference.state_paths.score import (
    StatePathScoringInputs,
    TrajectoryLogProbTerms,
    reconstruct_and_score_state_path,
)
from dynestyx.observation_missingness import MissingObservationStrategy
from dynestyx.simulation.base import _suspend_numpyro_plate_frames
from dynestyx.types import LatentStateResult


def _build_latent_state_result(
    *,
    prepared: _PreparedLatentPathRequest,
    assembled_state_path: AssembledStatePath | None,
    log_prob_terms: TrajectoryLogProbTerms | None,
) -> LatentStateResult:
    """Package one latent-path evaluation into the public result dataclass.

    ``LatentStateResult`` is the external object seen by callers. This helper
    translates the internal split between prepared inputs, assembled state
    paths, and log-probability terms into that public shape.
    """
    missing_obs_metadata = prepared.layout.missing_obs_metadata
    missing_obs_times = None
    missing_obs_coordinate_indices = None
    if log_prob_terms is None:
        if missing_obs_metadata is not None:
            missing_obs_times = missing_obs_metadata.missing_obs_times
            missing_obs_coordinate_indices = (
                missing_obs_metadata.missing_obs_coordinate_indices
            )
    else:
        missing_obs_times = log_prob_terms.missing_obs_times
        missing_obs_coordinate_indices = log_prob_terms.missing_obs_coordinate_indices

    return LatentStateResult(
        joint_log_prob=None
        if log_prob_terms is None
        else log_prob_terms.joint_log_prob,
        state_path_params=(
            None
            if assembled_state_path is None
            else assembled_state_path.state_path_params
        ),
        state_path_param_times=prepared.layout.state_path_param_times,
        state_path_param_coordinate_indices=(
            None
            if assembled_state_path is None
            else assembled_state_path.state_path_param_coordinate_indices
        ),
        state_path=(
            None if assembled_state_path is None else assembled_state_path.state_path
        ),
        state_path_times=(
            None
            if assembled_state_path is None
            else assembled_state_path.state_path_times
        ),
        missing_obs_values=(
            prepared.canonical_missing_obs_values
            if log_prob_terms is None
            else log_prob_terms.missing_obs_values
        ),
        missing_obs_times=missing_obs_times,
        missing_obs_coordinate_indices=missing_obs_coordinate_indices,
        completed_obs_values=(
            None if log_prob_terms is None else log_prob_terms.completed_obs_values
        ),
        state_dists=None,
    )


@dataclasses.dataclass(init=False)
class LatentPathBuilder(ObjectInterpretation, HandlesSelf):
    """Build latent path parameters and score ``log p(x, y | ...)``.

    Writing ``z = state_path_params`` and ``x = state_path = g(z)``, this
    handler constructs latent NumPyro sites for ``z``, reconstructs ``x``, and
    returns quantities associated with the joint density ``log p(x, y | ...)``.

    In pure-JAX terms, this handler owns the latent-state inference problem
    rather than the forward simulation problem. Compared with a simulator:

    - the simulator produces trajectories from the generative model,
    - ``LatentPathBuilder`` treats the latent trajectory as an inference object,
      represented by ``state_path_params``, and
    - NumPyro sites are only an outer registration layer placed on top of the
      pure-JAX reconstruction/scoring logic.
    """

    ode_diffeqsolve_settings: dict[str, Any] | None
    missing_observation_strategy: MissingObservationStrategy

    def __init__(
        self,
        ode_diffeqsolve_settings: dict[str, Any] | None = None,
        missing_observation_strategy: MissingObservationStrategy = "auto",
    ) -> None:
        """Initialize the latent-path builder.

        ``missing_observation_strategy`` controls whether partially missing
        observations are marginalized, augmented with explicit latent
        coordinates, or rejected. For ``DiracIdentityObservation`` models with
        missing data, latent-path inference resolves ``"auto"`` to augment
        semantics: the missing exact observations are treated as the free
        coordinates needed to reconstruct the state path.

        ``ode_diffeqsolve_settings`` is only used when the latent path for a
        deterministic continuous-time model must be reconstructed by solving an
        ODE from its initial condition.
        """
        self.ode_diffeqsolve_settings = ode_diffeqsolve_settings
        self.missing_observation_strategy = missing_observation_strategy

    def _sample_single(
        self,
        name: str,
        dynamics,
        *,
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_values_filled: Array | None = None,
        _obs_mask: Array | None = None,
        _obs_has_missing: bool | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        state_path_params: Array | None = None,
        missing_obs_values: Array | None = None,
    ) -> LatentStateResult:
        """Handle one non-plated latent-path request.

        The control flow is intentionally staged:

        1. prepare/canonicalize the latent request,
        2. perform an eager pure-JAX evaluation when possible, and
        3. run NumPyro side effects for latent sampling and output registration.

        This is the main single-trajectory implementation; plated requests are
        reduced to repeated calls to this method.
        """
        _validate_inference_supported_model_classes(dynamics)
        prepared = _prepare_latent_path_request(
            name=name,
            dynamics=dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_values_filled=_obs_values_filled,
            obs_mask=_obs_mask,
            obs_has_missing=_obs_has_missing,
            state_path_params=state_path_params,
            missing_obs_values=missing_obs_values,
            missing_observation_strategy=self.missing_observation_strategy,
        )
        assert obs_times is not None
        assert obs_values is not None
        scoring_inputs = StatePathScoringInputs(
            dynamics=dynamics,
            layout=prepared.layout,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_values_filled=prepared.obs_values_filled,
            obs_mask=prepared.obs_mask,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            missing_observation_strategy=self.missing_observation_strategy,
            ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
            canonical_state_path_params=prepared.canonical_state_path_params,
            canonical_missing_obs_values=prepared.canonical_missing_obs_values,
        )
        assembled_state_path, log_prob_terms = reconstruct_and_score_state_path(
            scoring_inputs
        )

        state_path_param_base_dist = dist.Normal(0.0, 1.0).expand(
            tuple(jnp.asarray(prepared.example_state_path_params).shape)
        )
        state_path_param_base_dist = state_path_param_base_dist.to_event(
            len(tuple(jnp.asarray(prepared.example_state_path_params).shape))
        )
        # Sample the state path parameters and missing observations.
        # Their log-probabilities are subtracted back out later.
        state_path_param_site = numpyro.sample(
            f"{name}_state_path_params",
            state_path_param_base_dist,
            obs=prepared.canonical_state_path_params,
        )
        missing_obs_site = None
        missing_obs_base_dist = None
        if prepared.example_missing_obs_values is not None:
            missing_obs_shape = tuple(
                jnp.asarray(prepared.example_missing_obs_values).shape
            )
            missing_obs_base_dist = (
                dist.Normal(0.0, 1.0)
                .expand(missing_obs_shape)
                .to_event(len(missing_obs_shape))
            )
            missing_obs_site = numpyro.sample(
                f"{name}_missing_obs_values",
                missing_obs_base_dist,
                obs=prepared.canonical_missing_obs_values,
            )

        sampled_state_path, sampled_log_prob_terms = reconstruct_and_score_state_path(
            scoring_inputs,
            state_path_params=state_path_param_site,
            missing_obs_values=missing_obs_site,
        )
        assert sampled_state_path is not None
        assert sampled_log_prob_terms is not None

        dummy_latent_log_prob = state_path_param_base_dist.log_prob(
            state_path_param_site
        )
        if missing_obs_site is not None and missing_obs_base_dist is not None:
            dummy_latent_log_prob = (
                dummy_latent_log_prob + missing_obs_base_dist.log_prob(missing_obs_site)
            )

        numpyro.factor(
            f"{name}_joint_log_prob_factor",
            sampled_log_prob_terms.joint_log_prob,
        )
        numpyro.factor(
            f"{name}_dummy_latent_log_prob_correction",
            -dummy_latent_log_prob,
        )
        numpyro.deterministic(
            f"{name}_state_path_param_times",
            sampled_state_path.state_path_param_times,
        )
        if sampled_state_path.state_path_param_coordinate_indices is not None:
            numpyro.deterministic(
                f"{name}_state_path_param_coordinate_indices",
                sampled_state_path.state_path_param_coordinate_indices,
            )
        numpyro.deterministic(f"{name}_state_path", sampled_state_path.state_path)
        numpyro.deterministic(
            f"{name}_state_path_times",
            sampled_state_path.state_path_times,
        )
        if sampled_log_prob_terms.missing_obs_times is not None:
            numpyro.deterministic(
                f"{name}_missing_obs_times",
                sampled_log_prob_terms.missing_obs_times,
            )
        if sampled_log_prob_terms.missing_obs_coordinate_indices is not None:
            numpyro.deterministic(
                f"{name}_missing_obs_coordinate_indices",
                sampled_log_prob_terms.missing_obs_coordinate_indices,
            )
        if sampled_log_prob_terms.completed_obs_values is not None:
            numpyro.deterministic(
                f"{name}_completed_obs_values",
                sampled_log_prob_terms.completed_obs_values,
            )
        numpyro.deterministic(
            f"{name}_joint_log_prob",
            sampled_log_prob_terms.joint_log_prob,
        )

        return _build_latent_state_result(
            prepared=prepared,
            assembled_state_path=assembled_state_path,
            log_prob_terms=log_prob_terms,
        )

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_values_filled: Array | None = None,
        _obs_mask: Array | None = None,
        _obs_has_missing: bool | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        state_path_params: Array | None = None,
        missing_obs_values: Array | None = None,
        _dsx_sample_mode: bool = False,
        **kwargs,
    ) -> LatentStateResult:
        """Interpret ``dsx.sample(...)`` for latent-path inference."""
        if not _dsx_sample_mode:
            raise ValueError(
                "LatentPathBuilder only supports dsx.sample(...) under NumPyro. "
                "Use dsx.log_prob(...) for pure-JAX trajectory scoring."
            )

        if not plate_shapes:
            return self._sample_single(
                name,
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                _obs_values_filled=_obs_values_filled,
                _obs_mask=_obs_mask,
                _obs_has_missing=_obs_has_missing,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                state_path_params=state_path_params,
                missing_obs_values=missing_obs_values,
            )

        member_specs = _plate_member_specs(
            name=name,
            dynamics=dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_values_filled=_obs_values_filled,
            obs_mask=_obs_mask,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            state_path_params=state_path_params,
            missing_obs_values=missing_obs_values,
        )

        member_results = []
        for member in member_specs:
            with _suspend_numpyro_plate_frames():
                member_results.append(
                    self._sample_single(
                        member.name,
                        member.dynamics,
                        obs_times=member.obs_times,
                        obs_values=member.obs_values,
                        _obs_values_filled=member.obs_values_filled,
                        _obs_mask=member.obs_mask,
                        _obs_has_missing=_obs_has_missing,
                        ctrl_times=member.ctrl_times,
                        ctrl_values=member.ctrl_values,
                        state_path_params=member.state_path_params,
                        missing_obs_values=member.missing_obs_values,
                    )
                )

        return LatentStateResult(
            joint_log_prob=_stack_member_attr(
                member_results, "joint_log_prob", plate_shapes
            ),
            state_path_params=_stack_member_attr(
                member_results, "state_path_params", plate_shapes
            ),
            state_path_param_times=_stack_member_attr(
                member_results, "state_path_param_times", plate_shapes
            ),
            state_path_param_coordinate_indices=_stack_member_attr(
                member_results,
                "state_path_param_coordinate_indices",
                plate_shapes,
            ),
            state_path=_stack_member_attr(member_results, "state_path", plate_shapes),
            state_path_times=_stack_member_attr(
                member_results, "state_path_times", plate_shapes
            ),
            missing_obs_values=_stack_member_attr(
                member_results, "missing_obs_values", plate_shapes
            ),
            missing_obs_times=_stack_member_attr(
                member_results, "missing_obs_times", plate_shapes
            ),
            missing_obs_coordinate_indices=_stack_member_attr(
                member_results, "missing_obs_coordinate_indices", plate_shapes
            ),
            completed_obs_values=_stack_member_attr(
                member_results, "completed_obs_values", plate_shapes
            ),
            state_dists=None,
        )


__all__ = ["LatentPathBuilder"]
