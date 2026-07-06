"""Latent-path handler for joint state + parameter inference."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.latent.state_path import (
    DiracLatentMetadata,
    assemble_dirac_state_path,
    assemble_state_path,
    canonicalize_dirac_state_path_params,
    canonicalize_state_path_params,
    fully_observed_dirac_state_path_param_metadata,
    infer_dirac_state_path_param_metadata,
    infer_state_path_param_times,
)
from dynestyx.inference.latent.trajectory_log_probs import (
    _compute_log_prob_terms_from_state_trajectory,
    compute_trajectory_log_prob_terms,
)
from dynestyx.inference.plate_utils import (
    _slice_array_for_plate_member,
    _slice_dist_for_plate_member,
)
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiracIdentityObservation,
)
from dynestyx.simulation.base import (
    _slice_tree_for_plate_member,
    _suspend_numpyro_plate_frames,
)
from dynestyx.types import LatentStateResult
from dynestyx.utils import _dist_has_plate_batch_dims


def _base_state_path_param_distribution(state_path_params: Array) -> dist.Distribution:
    """Return the dummy base distribution used for path-parameter construction."""
    param_shape = tuple(jnp.asarray(state_path_params).shape)
    base = dist.Normal(0.0, 1.0).expand(param_shape)
    return base.to_event(len(param_shape))


@dataclasses.dataclass
class LatentPathBuilder(ObjectInterpretation, HandlesSelf):
    """Build latent path parameters and score ``log p(x, y | ...)``.

    Writing ``z = state_path_params`` and ``x = g(z) = state_path``, this
    handler constructs or conditions on ``z``, reconstructs ``x``, and returns
    quantities associated with the joint density ``log p(x, y)``.

    The point of the ``state_path_params`` name is that ``z`` need not be the
    path itself. Today ``z`` may be the full discrete path, an ODE initial
    condition, or a compressed exact-observation representation. Longer-term,
    the same interface leaves room for surrogate path parameterizations, SDE
    matching constructions, and other non-identity ``x = g(z)`` maps.

    For partially missing `DiracIdentityObservation` models under traced
    NumPyro inference, precompute the compression metadata eagerly with
    ``prepare_dirac_state_path_metadata(...)`` and pass it here via
    ``dirac_state_path_metadata``.
    """

    ode_diffeqsolve_settings: dict[str, Any] | None = None
    dirac_state_path_metadata: DiracLatentMetadata | None = None

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
        _dsx_sample_mode: bool = False,
        **kwargs,
    ) -> LatentStateResult:
        if obs_times is None or obs_values is None:
            raise ValueError(
                "LatentPathBuilder requires obs_times and obs_values. "
                "It is an observation-consuming handler."
            )

        is_dirac_observation = isinstance(
            dynamics.observation_model, DiracIdentityObservation
        )
        is_ode = isinstance(
            dynamics.state_evolution, DeterministicContinuousTimeStateEvolution
        )
        use_dirac_compression = (
            is_dirac_observation
            and not is_ode
            and _obs_values_filled is not None
            and (_obs_has_missing is False or _obs_mask is not None)
        )

        dirac_metadata = None
        if use_dirac_compression:
            if self.dirac_state_path_metadata is not None:
                dirac_metadata = self.dirac_state_path_metadata
                if _obs_values_filled is not None and (
                    dirac_metadata.state_shape
                    != tuple(jnp.asarray(_obs_values_filled).shape)
                ):
                    raise ValueError(
                        "dirac_state_path_metadata.state_shape does not match the "
                        "observed data shape for this LatentPathBuilder call."
                    )
            elif _obs_mask is not None:
                dirac_metadata = infer_dirac_state_path_param_metadata(
                    dynamics,
                    obs_times=obs_times,
                    obs_mask=_obs_mask,
                )
            else:
                assert _obs_values_filled is not None
                dirac_metadata = fully_observed_dirac_state_path_param_metadata(
                    obs_times=obs_times,
                    state_shape=tuple(jnp.asarray(_obs_values_filled).shape),
                )
        dirac_obs_values_filled_array: Array | None = None
        dirac_obs_mask_array: Array | None = None
        if dirac_metadata is not None:
            assert _obs_values_filled is not None
            assert _obs_mask is not None
            dirac_obs_values_filled_array = _obs_values_filled
            dirac_obs_mask_array = _obs_mask
        state_path_param_times = (
            dirac_metadata.state_path_param_times
            if dirac_metadata is not None
            else infer_state_path_param_times(dynamics, obs_times=obs_times)
        )

        if state_path_params is None and not _dsx_sample_mode:
            raise ValueError(
                "state_path_params must be provided when using dsx.condition with "
                "LatentPathBuilder. Use dsx.sample under NumPyro to sample them."
            )

        assembled = None
        terms = None
        canonical_state_path_params = None
        if state_path_params is not None:
            if dirac_metadata is not None:
                assert dirac_obs_values_filled_array is not None
                assert dirac_obs_mask_array is not None
                canonical_state_path_params = canonicalize_dirac_state_path_params(
                    state_path_params,
                    n_state_path_params=dirac_metadata.free_flat_indices.shape[0],
                )
                assembled = assemble_dirac_state_path(
                    state_path_params=canonical_state_path_params,
                    latent_metadata=dirac_metadata,
                    obs_times=obs_times,
                    obs_values_filled=dirac_obs_values_filled_array,
                )
                terms = _compute_log_prob_terms_from_state_trajectory(
                    dynamics,
                    state_path=assembled.state_path,
                    state_path_times=assembled.state_path_times,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    obs_values_filled=dirac_obs_values_filled_array,
                    obs_mask=dirac_obs_mask_array,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    observations_are_exact_constraints=True,
                )
            else:
                canonical_state_path_params = canonicalize_state_path_params(
                    dynamics,
                    state_path_params,
                    n_times=state_path_param_times.shape[0],
                )
                assembled = assemble_state_path(
                    dynamics,
                    state_path_params=canonical_state_path_params,
                    state_path_param_times=state_path_param_times,
                    obs_times=obs_times,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
                )
                terms = compute_trajectory_log_prob_terms(
                    dynamics,
                    state_path_params=canonical_state_path_params,
                    state_path_param_times=state_path_param_times,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    obs_values_filled=_obs_values_filled,
                    obs_mask=_obs_mask,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
                )

        example_state_path_params = (
            canonical_state_path_params
            if canonical_state_path_params is not None
            else (
                jnp.zeros((dirac_metadata.free_flat_indices.shape[0],))
                if dirac_metadata is not None
                else canonicalize_state_path_params(
                    dynamics,
                    jnp.zeros(
                        (
                            state_path_param_times.shape[0],
                            *dynamics.initial_condition.event_shape,
                        )
                    ),
                    n_times=state_path_param_times.shape[0],
                )
            )
        )
        base_dist = _base_state_path_param_distribution(example_state_path_params)

        def _register(site_name: str) -> None:
            path_param_site = numpyro.sample(
                f"{site_name}_state_path_params",
                base_dist,
                obs=canonical_state_path_params,
            )
            if dirac_metadata is not None:
                assert dirac_obs_values_filled_array is not None
                assert dirac_obs_mask_array is not None
                assembled_now = assemble_dirac_state_path(
                    state_path_params=path_param_site,
                    latent_metadata=dirac_metadata,
                    obs_times=obs_times,
                    obs_values_filled=dirac_obs_values_filled_array,
                )
                terms_now = _compute_log_prob_terms_from_state_trajectory(
                    dynamics,
                    state_path=assembled_now.state_path,
                    state_path_times=assembled_now.state_path_times,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    obs_values_filled=dirac_obs_values_filled_array,
                    obs_mask=dirac_obs_mask_array,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    observations_are_exact_constraints=True,
                )
            else:
                assembled_now = assemble_state_path(
                    dynamics,
                    state_path_params=path_param_site,
                    state_path_param_times=state_path_param_times,
                    obs_times=obs_times,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
                )
                terms_now = compute_trajectory_log_prob_terms(
                    dynamics,
                    state_path_params=path_param_site,
                    state_path_param_times=state_path_param_times,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    obs_values_filled=_obs_values_filled,
                    obs_mask=_obs_mask,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
                )
            numpyro.factor(
                f"{site_name}_state_path_params_lp",
                terms_now.joint_log_prob - base_dist.log_prob(path_param_site),
            )
            numpyro.deterministic(
                f"{site_name}_state_path_param_times",
                state_path_param_times,
            )
            if assembled_now.state_path_param_coordinate_indices is not None:
                numpyro.deterministic(
                    f"{site_name}_state_path_param_coordinate_indices",
                    assembled_now.state_path_param_coordinate_indices,
                )
            numpyro.deterministic(f"{site_name}_state_path", assembled_now.state_path)
            numpyro.deterministic(
                f"{site_name}_state_path_times", assembled_now.state_path_times
            )
            numpyro.deterministic(
                f"{site_name}_joint_log_prob", terms_now.joint_log_prob
            )

        return LatentStateResult(
            joint_log_prob=None if terms is None else terms.joint_log_prob,
            state_path_params=(
                None if assembled is None else assembled.state_path_params
            ),
            state_path_param_times=state_path_param_times,
            state_path_param_coordinate_indices=(
                None
                if assembled is None
                else assembled.state_path_param_coordinate_indices
            ),
            state_path=None if assembled is None else assembled.state_path,
            state_path_times=None if assembled is None else assembled.state_path_times,
            state_dists=None,
            _register_numpyro_sites=_register,
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
        _dsx_sample_mode: bool = False,
        **kwargs,
    ) -> LatentStateResult:
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
                _dsx_sample_mode=_dsx_sample_mode,
                **kwargs,
            )

        if not _dsx_sample_mode:
            raise NotImplementedError(
                "LatentPathBuilder does not yet support dsx.condition batched latent-path "
                "construction. Use dsx.sample under NumPyro, or remove dsx.plate."
            )

        member_specs = []
        for plate_idx in itertools.product(*[range(s) for s in plate_shapes]):
            member_name = f"{name}_p{'_'.join(str(i) for i in plate_idx)}"
            member_dynamics = _slice_tree_for_plate_member(
                dynamics, plate_shapes, plate_idx
            )
            if _dist_has_plate_batch_dims(dynamics.initial_condition, plate_shapes):
                member_initial_condition = _slice_dist_for_plate_member(
                    dynamics.initial_condition, plate_shapes, plate_idx
                )
                member_dynamics = eqx.tree_at(
                    lambda m: m.initial_condition,
                    member_dynamics,
                    member_initial_condition,
                    is_leaf=lambda x: x is None,
                )

            member_specs.append(
                (
                    member_name,
                    member_dynamics,
                    _slice_array_for_plate_member(obs_times, plate_shapes, plate_idx),
                    _slice_array_for_plate_member(obs_values, plate_shapes, plate_idx),
                    _slice_array_for_plate_member(
                        _obs_values_filled, plate_shapes, plate_idx
                    ),
                    _slice_array_for_plate_member(_obs_mask, plate_shapes, plate_idx),
                    _slice_array_for_plate_member(ctrl_times, plate_shapes, plate_idx),
                    _slice_array_for_plate_member(ctrl_values, plate_shapes, plate_idx),
                    _slice_array_for_plate_member(
                        state_path_params, plate_shapes, plate_idx
                    ),
                )
            )

        member_results = []
        for (
            member_name,
            member_dynamics,
            member_obs_times,
            member_obs_values,
            member_obs_values_filled,
            member_obs_mask,
            member_ctrl_times,
            member_ctrl_values,
            member_state_path_params,
        ) in member_specs:
            with _suspend_numpyro_plate_frames():
                member_results.append(
                    self._sample_single(
                        member_name,
                        member_dynamics,
                        obs_times=member_obs_times,
                        obs_values=member_obs_values,
                        _obs_values_filled=member_obs_values_filled,
                        _obs_mask=member_obs_mask,
                        _obs_has_missing=_obs_has_missing,
                        ctrl_times=member_ctrl_times,
                        ctrl_values=member_ctrl_values,
                        state_path_params=member_state_path_params,
                        _dsx_sample_mode=True,
                        **kwargs,
                    )
                )

        def _stack(attr: str):
            values = [getattr(result, attr) for result in member_results]
            if any(value is None for value in values):
                return None
            return jnp.stack([jnp.asarray(value) for value in values]).reshape(
                *plate_shapes, *jnp.asarray(values[0]).shape
            )

        def _register(_site_name: str) -> None:
            for (member_name, *_), member_result in zip(
                member_specs, member_results, strict=True
            ):
                register = getattr(member_result, "_register_numpyro_sites", None)
                if callable(register):
                    with _suspend_numpyro_plate_frames():
                        register(member_name)

        return LatentStateResult(
            joint_log_prob=_stack("joint_log_prob"),
            state_path_params=_stack("state_path_params"),
            state_path_param_times=_stack("state_path_param_times"),
            state_path_param_coordinate_indices=_stack(
                "state_path_param_coordinate_indices"
            ),
            state_path=_stack("state_path"),
            state_path_times=_stack("state_path_times"),
            state_dists=None,
            _register_numpyro_sites=_register,
        )
