"""Latent-path handler for joint state + parameter inference.

This module is the orchestration layer for latent-path inference. It does not
define the latent parameterization ``z`` or the scoring rules itself; instead it
coordinates three steps:

1. prepare a concrete :class:`StatePathParameterization`,
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

from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.checkers import _validate_inference_supported_model_classes
from dynestyx.inference.latent._numpyro import build_latent_path_site_registrar
from dynestyx.inference.latent.log_prob import (
    TrajectoryLogProbTerms,
    compute_state_path_log_prob_terms,
)
from dynestyx.inference.latent.parameterization import (
    AssembledStatePath,
    StatePathParameterization,
)
from dynestyx.inference.latent.plate import (
    _plate_member_specs,
    _stack_member_attr,
)
from dynestyx.inference.latent.prepare import (
    _prepare_latent_path_request,
    _PreparedLatentPathRequest,
)
from dynestyx.observation_missingness import MissingObservationStrategy
from dynestyx.simulation.base import _suspend_numpyro_plate_frames
from dynestyx.types import LatentStateResult


def _evaluate_latent_path_request(
    *,
    dynamics,
    prepared: _PreparedLatentPathRequest,
    obs_times: Array,
    obs_values: Array,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
    latent_observation_mode: MissingObservationStrategy,
    ode_diffeqsolve_settings: dict[str, Any] | None,
    state_path_params: Array | None = None,
    missing_obs_values: Array | None = None,
) -> tuple[AssembledStatePath | None, TrajectoryLogProbTerms | None]:
    """Reconstruct the state path and score ``log p(x, y | ...)``.

    This is the core pure-JAX evaluation step. Given concrete latent values,
    it first reconstructs the full state path ``x = g(z)`` and then computes
    the joint trajectory score

    ``log p(x_0) + sum_t log p(x_{t+1} | x_t) + sum_t log p(y_t | x_t)``.

    Returning ``None`` is only possible when no concrete ``state_path_params``
    are available yet, which occurs on the ``dsx.sample(...)`` path before
    NumPyro has sampled the latent sites.
    """
    active_state_path_params = (
        prepared.canonical_state_path_params
        if state_path_params is None
        else state_path_params
    )
    if active_state_path_params is None:
        return None, None

    active_missing_obs_values = (
        prepared.canonical_missing_obs_values
        if missing_obs_values is None
        else missing_obs_values
    )
    assembled_state_path = prepared.parameterization.assemble_state_path(
        dynamics,
        state_path_params=active_state_path_params,
        obs_times=obs_times,
        obs_values_filled=prepared.obs_values_filled,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        ode_diffeqsolve_settings=ode_diffeqsolve_settings,
    )
    log_prob_terms = compute_state_path_log_prob_terms(
        dynamics,
        state_path=assembled_state_path.state_path,
        state_path_times=assembled_state_path.state_path_times,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_values_filled=prepared.obs_values_filled,
        obs_mask=prepared.obs_mask,
        missing_observation_strategy=latent_observation_mode,
        missing_obs_values=active_missing_obs_values,
        missing_obs_metadata=prepared.parameterization.missing_obs_metadata,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        observations_are_exact_constraints=(
            prepared.parameterization.dirac_metadata is not None
        ),
    )
    return assembled_state_path, log_prob_terms


def _build_latent_state_result(
    *,
    prepared: _PreparedLatentPathRequest,
    assembled_state_path: AssembledStatePath | None,
    log_prob_terms: TrajectoryLogProbTerms | None,
    register_numpyro_sites,
) -> LatentStateResult:
    """Package one latent-path evaluation into the public result dataclass.

    ``LatentStateResult`` is the external object seen by callers. This helper
    translates the internal split between prepared inputs, assembled state
    paths, and log-probability terms into that public shape, while also
    attaching the deferred ``_register_numpyro_sites`` callback.
    """
    missing_obs_metadata = prepared.parameterization.missing_obs_metadata
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
        state_path_param_times=prepared.parameterization.state_path_param_times,
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
        _register_numpyro_sites=register_numpyro_sites,
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
    latent_observation_mode: MissingObservationStrategy

    def __init__(
        self,
        ode_diffeqsolve_settings: dict[str, Any] | None = None,
        latent_observation_mode: MissingObservationStrategy = "auto",
        missing_observation_strategy: MissingObservationStrategy | None = None,
    ) -> None:
        """Initialize the latent-path builder.

        ``latent_observation_mode`` controls whether partially missing
        observations are marginalized, augmented with explicit latent
        coordinates, or rejected. ``missing_observation_strategy`` remains as a
        backward-compatible alias for the same setting.

        ``ode_diffeqsolve_settings`` is only used when the latent path for a
        deterministic continuous-time model must be reconstructed by solving an
        ODE from its initial condition.
        """
        if missing_observation_strategy is not None:
            if latent_observation_mode != "auto":
                raise ValueError(
                    "Pass only one of latent_observation_mode or "
                    "missing_observation_strategy."
                )
            latent_observation_mode = missing_observation_strategy
        self.ode_diffeqsolve_settings = ode_diffeqsolve_settings
        self.latent_observation_mode = latent_observation_mode

    @property
    def missing_observation_strategy(self) -> MissingObservationStrategy:
        """Backward-compatible alias for ``latent_observation_mode``."""
        return self.latent_observation_mode

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
        latent_path_layout: StatePathParameterization | None = None,
        state_path_params: Array | None = None,
        missing_obs_values: Array | None = None,
        **kwargs,
    ) -> LatentStateResult:
        """Handle one non-plated latent-path request.

        The control flow is intentionally staged:

        1. prepare/canonicalize the latent request,
        2. perform an eager pure-JAX evaluation when possible, and
        3. build a deferred NumPyro registration callback that can either reuse
           the eager result or recompute it from sampled latent values.

        This is the main single-trajectory implementation; plated requests are
        reduced to repeated calls to this method.
        """
        _validate_inference_supported_model_classes(dynamics)
        prepared = _prepare_latent_path_request(
            dynamics=dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_values_filled=_obs_values_filled,
            obs_mask=_obs_mask,
            obs_has_missing=_obs_has_missing,
            latent_path_layout=latent_path_layout,
            state_path_params=state_path_params,
            missing_obs_values=missing_obs_values,
            latent_observation_mode=self.latent_observation_mode,
        )
        assert obs_times is not None
        assert obs_values is not None
        assembled_state_path, log_prob_terms = _evaluate_latent_path_request(
            dynamics=dynamics,
            prepared=prepared,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            latent_observation_mode=self.latent_observation_mode,
            ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
        )

        def _evaluate_latent_values(
            state_path_params_value: Array,
            missing_obs_values_value: Array | None,
        ) -> tuple[AssembledStatePath, TrajectoryLogProbTerms]:
            assembled_now, terms_now = _evaluate_latent_path_request(
                dynamics=dynamics,
                prepared=prepared,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                latent_observation_mode=self.latent_observation_mode,
                ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
                state_path_params=state_path_params_value,
                missing_obs_values=missing_obs_values_value,
            )
            assert assembled_now is not None
            assert terms_now is not None
            return assembled_now, terms_now

        register_numpyro_sites = build_latent_path_site_registrar(
            canonical_state_path_params=prepared.canonical_state_path_params,
            canonical_missing_obs_values=prepared.canonical_missing_obs_values,
            example_state_path_params=prepared.example_state_path_params,
            example_missing_obs_values=prepared.example_missing_obs_values,
            eager_assembled_state_path=assembled_state_path,
            eager_log_prob_terms=log_prob_terms,
            evaluate_latent_values=_evaluate_latent_values,
        )

        return _build_latent_state_result(
            prepared=prepared,
            assembled_state_path=assembled_state_path,
            log_prob_terms=log_prob_terms,
            register_numpyro_sites=register_numpyro_sites,
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
        latent_path_layout: StatePathParameterization | None = None,
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
                latent_path_layout=latent_path_layout,
                state_path_params=state_path_params,
                missing_obs_values=missing_obs_values,
                **kwargs,
            )

        if latent_path_layout is not None:
            raise NotImplementedError(
                "Plated LatentPathBuilder requests do not yet support an explicit "
                "latent_path_layout override."
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
                        **kwargs,
                    )
                )

        def _register(_site_name: str) -> None:
            for member, member_result in zip(member_specs, member_results, strict=True):
                register = getattr(member_result, "_register_numpyro_sites", None)
                if callable(register):
                    with _suspend_numpyro_plate_frames():
                        register(member.name)

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
            _register_numpyro_sites=_register,
        )


__all__ = ["LatentPathBuilder"]
