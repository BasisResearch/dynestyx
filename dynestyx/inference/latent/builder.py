"""NumPyro-facing explicit latent-path inference."""

from __future__ import annotations

import itertools
from typing import Any, cast

import equinox as eqx
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax.core import Tracer
from jaxtyping import Array, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.checkers import _validate_inference_supported_model_classes
from dynestyx.inference.posterior_rollout import (
    _final_times_for_rollout,
    _validate_future_only_predict_times,
)
from dynestyx.inference.state_paths.reconstruct import (
    infer_state_path_param_times,
    reconstruct_state_path,
    reconstruct_state_path_from_exact_observations,
    validate_state_path_params,
)
from dynestyx.inference.state_paths.score import (
    _gather_by_exact_time,
    compute_state_path_log_prob,
)
from dynestyx.inference.utils.distribution_utils import (
    _ForwardSimulationImproperUniform,
)
from dynestyx.inference.utils.plate_utils import (
    _slice_array_for_plate_member,
    _slice_dynamics_for_plate_member,
    _stack_optional_member_values,
    _suspend_numpyro_plate_frames,
)
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiracIdentityObservation,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
    assemble_completed_observations,
    prepare_missing_observation_metadata,
    resolve_missing_observation_strategy,
    validate_missing_obs_values,
)
from dynestyx.simulation.discrete import _sample_discrete_state_path
from dynestyx.simulation.utils import _sample_observation_path
from dynestyx.types import LatentStateResult
from dynestyx.utils import _build_control_path_eval

_MISSING_OBSERVATION_METADATA_CACHE: dict[
    tuple[str, str, tuple[int, ...], MissingObservationStrategy],
    MissingObservationMetadata,
] = {}


def _resolve_missing_observation_metadata(
    *,
    name: str,
    role: str,
    dynamics,
    obs_times: Array,
    obs_values: Array,
    obs_mask: Array,
    strategy: MissingObservationStrategy,
) -> MissingObservationMetadata:
    """Resolve missing-observation metadata.

    Concrete calls compute and cache it; traced calls retrieve it from the cache.
    """
    cache_key = (name, role, tuple(obs_values.shape), strategy)
    if any(isinstance(value, Tracer) for value in (obs_times, obs_values, obs_mask)):
        metadata = _MISSING_OBSERVATION_METADATA_CACHE.get(cache_key)
        if metadata is None:
            raise ValueError(
                "LatentPathBuilder encountered traced observation missingness before "
                "a compatible concrete model execution. Run the model once with "
                "concrete obs_times/obs_values before traced replay."
            )
        if metadata.observation_shape != tuple(obs_values.shape):
            raise ValueError(
                "Cached LatentPathBuilder missingness metadata does not match the "
                "current observation shape."
            )
        return metadata

    metadata = prepare_missing_observation_metadata(
        dynamics,
        obs_times=obs_times,
        obs_mask=obs_mask,
    )
    _MISSING_OBSERVATION_METADATA_CACHE[cache_key] = metadata
    return metadata


def _sample_missing_observation_prior(
    dynamics,
    state_path: Array,
    state_path_times: Array,
    obs_times: Array,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
    missing_flat_indices: Array,
    key: Array,
) -> Array:
    """Sample missing observations conditional on the assembled state path."""
    if missing_flat_indices.shape[0] == 0:
        return jnp.zeros((0,), dtype=jnp.asarray(state_path).dtype)

    states_at_observations = _gather_by_exact_time(
        state_path,
        state_path_times,
        obs_times,
        value_name="state_path",
    )
    control_path_eval = _build_control_path_eval(ctrl_times, ctrl_values, obs_times)
    dense_observations = _sample_observation_path(
        dynamics,
        states=states_at_observations,
        times=obs_times,
        rng_key=key,
        control_path_eval=control_path_eval,
    )
    return jnp.reshape(dense_observations, (-1,))[missing_flat_indices]


def _build_state_path_distributions(
    dynamics,
    state_path: Array,
) -> list[dist.Distribution]:
    """Wrap each inferred state in a Delta distribution for rollout forwarding."""
    event_dim = len(dynamics.initial_condition.event_shape)
    time_major_state_path = jnp.moveaxis(jnp.asarray(state_path), -(event_dim + 1), 0)
    return [
        dist.Delta(state_t, event_dim=event_dim) for state_t in time_major_state_path
    ]


class LatentPathBuilder(ObjectInterpretation, HandlesSelf):
    """Build explicit latent paths and score ``log p(x, y | ...)``."""

    def __init__(
        self,
        ode_diffeqsolve_settings: dict[str, Any] | None = None,
        missing_observation_strategy: MissingObservationStrategy = "auto",
        chunk_size: int | None = None,
    ) -> None:
        """Initialize explicit latent-path inference.

        ``ode_diffeqsolve_settings`` configures ODE reconstruction from a
        sampled initial condition. ``missing_observation_strategy`` selects
        marginalization, explicit augmentation, automatic selection, or an
        error for missing observations. Missing exact identity observations
        support automatic selection and explicit augmentation only.

        ``chunk_size`` limits the number of transition or observation terms
        evaluated in each ``lax.scan`` chunk. By default, all terms are
        evaluated in one ``vmap``.
        """
        self.ode_diffeqsolve_settings = ode_diffeqsolve_settings
        self.missing_observation_strategy = missing_observation_strategy
        self.chunk_size = chunk_size

    def _sample_single(
        self,
        name: str,
        dynamics,
        *,
        obs_times: Array | None,
        obs_values: Array | None,
        obs_values_filled: Array | None,
        obs_mask: Array | None,
        ctrl_times: Array | None,
        ctrl_values: Array | None,
        state_path_params: Array | None,
        missing_obs_values: Array | None,
    ) -> LatentStateResult:
        _validate_inference_supported_model_classes(dynamics)
        if isinstance(
            dynamics.state_evolution,
            StochasticContinuousTimeStateEvolution,
        ):
            raise ValueError(
                "Latent-state assembly does not yet support native SDE models. "
                "Please discretize the model first."
            )
        if obs_times is None or obs_values is None:
            raise ValueError(
                "LatentPathBuilder requires obs_times and obs_values. "
                "It is an observation-consuming handler."
            )
        if self.missing_observation_strategy not in (
            "auto",
            "marginalize",
            "augment",
            "error",
        ):
            raise ValueError(
                "missing_observation_strategy must be one of 'auto', "
                "'marginalize', 'augment', or 'error'."
            )

        if obs_values_filled is None or obs_mask is None:
            raise ValueError(
                "LatentPathBuilder requires observation views prepared by "
                "dsx.sample(...)."
            )

        metadata = _resolve_missing_observation_metadata(
            name=name,
            role="observations",
            dynamics=dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_mask=obs_mask,
            strategy=self.missing_observation_strategy,
        )
        exact_observations = isinstance(
            dynamics.observation_model,
            DiracIdentityObservation,
        )

        if (
            exact_observations
            and metadata.has_missing
            and (self.missing_observation_strategy in ("marginalize", "error"))
        ):
            raise ValueError(
                "DiracIdentityObservation missingness in latent-path inference "
                "supports only augment semantics. Use "
                "missing_observation_strategy='auto' or 'augment'."
            )

        observation_dim = 1 if obs_values.ndim == 1 else obs_values.shape[-1]
        use_observation_augmentation = False
        if not exact_observations:
            use_observation_augmentation, _ = resolve_missing_observation_strategy(
                dynamics,
                observation_dim=observation_dim,
                has_missing=metadata.has_missing,
                has_partial_missing=metadata.has_partial_missing,
                requested_strategy=self.missing_observation_strategy,
            )

        state_path_param_coordinate_indices = None
        state_sample_transform = None
        if exact_observations:
            state_path_param_times = metadata.missing_obs_times
            state_path_param_coordinate_indices = (
                metadata.missing_obs_coordinate_indices
            )
            state_event_shape = (metadata.missing_flat_indices.shape[0],)
            validated_state_path_params = (
                None
                if state_path_params is None
                else validate_missing_obs_values(
                    state_path_params,
                    n_missing_obs=metadata.missing_flat_indices.shape[0],
                )
            )
            state_forward_sampler = eqx.Partial(
                _sample_discrete_state_path,
                dynamics=dynamics,
                times=obs_times,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )
            state_sample_transform = eqx.Partial(
                jnp.take,
                indices=metadata.missing_flat_indices,
            )
        else:
            state_path_param_times = infer_state_path_param_times(
                dynamics,
                obs_times=obs_times,
            )
            state_event_shape = (
                state_path_param_times.shape[0],
                *dynamics.initial_condition.event_shape,
            )
            validated_state_path_params = (
                None
                if state_path_params is None
                else validate_state_path_params(
                    dynamics,
                    state_path_params,
                    n_times=state_path_param_times.shape[0],
                )
            )
            state_forward_sampler = (
                dynamics.initial_condition
                if isinstance(
                    dynamics.state_evolution,
                    DeterministicContinuousTimeStateEvolution,
                )
                else eqx.Partial(
                    _sample_discrete_state_path,
                    dynamics=dynamics,
                    times=obs_times,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                )
            )

        state_path_param_site = numpyro.sample(
            f"{name}_state_path_params",
            _ForwardSimulationImproperUniform(
                state_forward_sampler,
                event_shape=state_event_shape,
                sample_transform=state_sample_transform,
            ),
            obs=validated_state_path_params,
        )

        if exact_observations:
            validated_state_path_params, state_path, state_path_times = (
                reconstruct_state_path_from_exact_observations(
                    state_path_params=state_path_param_site,
                    latent_metadata=metadata,
                    obs_times=obs_times,
                    obs_values_filled=obs_values_filled,
                )
            )
        else:
            validated_state_path_params, state_path, state_path_times = (
                reconstruct_state_path(
                    dynamics,
                    state_path_params=state_path_param_site,
                    state_path_param_times=state_path_param_times,
                    obs_times=obs_times,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    ode_diffeqsolve_settings=self.ode_diffeqsolve_settings,
                )
            )

        missing_obs_site = None
        missing_obs_times = None
        missing_obs_coordinate_indices = None
        completed_obs_values = state_path if exact_observations else None
        if use_observation_augmentation:
            n_missing_obs = metadata.missing_flat_indices.shape[0]
            validated_missing_obs_values = (
                None
                if missing_obs_values is None
                else validate_missing_obs_values(
                    missing_obs_values,
                    n_missing_obs=n_missing_obs,
                )
            )
            observation_forward_sampler = eqx.Partial(
                _sample_missing_observation_prior,
                dynamics,
                state_path,
                state_path_times,
                obs_times,
                ctrl_times,
                ctrl_values,
                metadata.missing_flat_indices,
            )
            missing_obs_site = numpyro.sample(
                f"{name}_missing_obs_values",
                _ForwardSimulationImproperUniform(
                    observation_forward_sampler,
                    event_shape=(n_missing_obs,),
                ),
                obs=validated_missing_obs_values,
            )
            completed_obs_values = assemble_completed_observations(
                obs_values_filled=obs_values_filled,
                missing_obs_values=missing_obs_site,
                missing_obs_metadata=metadata,
            )
            missing_obs_times = metadata.missing_obs_times
            missing_obs_coordinate_indices = metadata.missing_obs_coordinate_indices
        elif missing_obs_values is not None:
            raise ValueError(
                "missing_obs_values was provided, but this request does not use "
                "explicit missing-observation augmentation."
            )

        joint_log_prob = compute_state_path_log_prob(
            dynamics,
            state_path=state_path,
            state_path_times=state_path_times,
            obs_times=obs_times,
            obs_values=obs_values,
            obs_values_filled=obs_values_filled,
            obs_mask=obs_mask,
            missing_observation_strategy=self.missing_observation_strategy,
            missing_obs_values=missing_obs_site,
            missing_obs_metadata=(metadata if use_observation_augmentation else None),
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            chunk_size=self.chunk_size,
            observations_are_exact_constraints=exact_observations,
        )

        numpyro.factor(f"{name}_joint_log_prob_factor", joint_log_prob)
        numpyro.deterministic(
            f"{name}_state_path_param_times",
            state_path_param_times,
        )
        if state_path_param_coordinate_indices is not None:
            numpyro.deterministic(
                f"{name}_state_path_param_coordinate_indices",
                state_path_param_coordinate_indices,
            )
        numpyro.deterministic(f"{name}_state_path", state_path)
        numpyro.deterministic(f"{name}_state_path_times", state_path_times)
        if missing_obs_times is not None:
            numpyro.deterministic(f"{name}_missing_obs_times", missing_obs_times)
        if missing_obs_coordinate_indices is not None:
            numpyro.deterministic(
                f"{name}_missing_obs_coordinate_indices",
                missing_obs_coordinate_indices,
            )
        if completed_obs_values is not None:
            numpyro.deterministic(
                f"{name}_completed_obs_values",
                completed_obs_values,
            )
        numpyro.deterministic(f"{name}_joint_log_prob", joint_log_prob)

        return LatentStateResult(
            joint_log_prob=joint_log_prob,
            state_path_params=validated_state_path_params,
            state_path_param_times=state_path_param_times,
            state_path_param_coordinate_indices=state_path_param_coordinate_indices,
            state_path=state_path,
            state_path_times=state_path_times,
            missing_obs_values=missing_obs_site,
            missing_obs_times=missing_obs_times,
            missing_obs_coordinate_indices=missing_obs_coordinate_indices,
            completed_obs_values=completed_obs_values,
            state_dists=_build_state_path_distributions(dynamics, state_path),
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
        predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
        state_path_params: Array | None = None,
        missing_obs_values: Array | None = None,
        _dsx_sample_mode: bool = False,
        **kwargs,
    ) -> LatentStateResult:
        if not _dsx_sample_mode:
            raise ValueError(
                "LatentPathBuilder only supports dsx.sample(...) under NumPyro. "
                "Use dsx.log_prob(...) for pure-JAX trajectory scoring."
            )

        if not plate_shapes:
            result = self._sample_single(
                name,
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                obs_values_filled=_obs_values_filled,
                obs_mask=_obs_mask,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                state_path_params=state_path_params,
                missing_obs_values=missing_obs_values,
            )
        else:
            member_results: list[LatentStateResult] = []
            for plate_idx in itertools.product(*[range(size) for size in plate_shapes]):
                member_name = f"{name}_p{'_'.join(str(i) for i in plate_idx)}"
                with _suspend_numpyro_plate_frames():
                    member_results.append(
                        self._sample_single(
                            member_name,
                            _slice_dynamics_for_plate_member(
                                dynamics,
                                plate_shapes,
                                plate_idx,
                            ),
                            obs_times=_slice_array_for_plate_member(
                                obs_times, plate_shapes, plate_idx
                            ),
                            obs_values=_slice_array_for_plate_member(
                                obs_values, plate_shapes, plate_idx
                            ),
                            obs_values_filled=_slice_array_for_plate_member(
                                _obs_values_filled, plate_shapes, plate_idx
                            ),
                            obs_mask=_slice_array_for_plate_member(
                                _obs_mask, plate_shapes, plate_idx
                            ),
                            ctrl_times=_slice_array_for_plate_member(
                                ctrl_times, plate_shapes, plate_idx
                            ),
                            ctrl_values=_slice_array_for_plate_member(
                                ctrl_values, plate_shapes, plate_idx
                            ),
                            state_path_params=_slice_array_for_plate_member(
                                state_path_params, plate_shapes, plate_idx
                            ),
                            missing_obs_values=_slice_array_for_plate_member(
                                missing_obs_values, plate_shapes, plate_idx
                            ),
                        )
                    )

            stacked_member_values = {
                attr: _stack_optional_member_values(
                    [getattr(member, attr) for member in member_results],
                    plate_shapes,
                )
                for attr in (
                    "joint_log_prob",
                    "state_path_params",
                    "state_path_param_times",
                    "state_path_param_coordinate_indices",
                    "state_path",
                    "state_path_times",
                    "missing_obs_values",
                    "missing_obs_times",
                    "missing_obs_coordinate_indices",
                    "completed_obs_values",
                )
            }
            state_path = stacked_member_values["state_path"]
            result = LatentStateResult(
                joint_log_prob=stacked_member_values["joint_log_prob"],
                state_path_params=stacked_member_values["state_path_params"],
                state_path_param_times=stacked_member_values["state_path_param_times"],
                state_path_param_coordinate_indices=stacked_member_values[
                    "state_path_param_coordinate_indices"
                ],
                state_path=state_path,
                state_path_times=stacked_member_values["state_path_times"],
                missing_obs_values=stacked_member_values["missing_obs_values"],
                missing_obs_times=stacked_member_values["missing_obs_times"],
                missing_obs_coordinate_indices=stacked_member_values[
                    "missing_obs_coordinate_indices"
                ],
                completed_obs_values=stacked_member_values["completed_obs_values"],
                state_dists=(
                    None
                    if state_path is None
                    else _build_state_path_distributions(dynamics, state_path)
                ),
            )

        predict_times = _validate_future_only_predict_times(
            predict_times,
            cast(Array | None, result.state_path_times),
            error_message=(
                "LatentPathBuilder rollout only supports predict_times >= "
                "max(state_path_times); in-window latent-path predictions are "
                "not implemented yet."
            ),
        )
        filtered_times = None
        filtered_dists = None
        posterior_rollout_final_only = False
        smoothed_times = result.state_path_times
        smoothed_dists = result.state_dists
        if predict_times is not None and smoothed_dists:
            assert result.state_path_times is not None
            filtered_times = _final_times_for_rollout(
                cast(Array, result.state_path_times)
            )
            filtered_dists = [smoothed_dists[-1]]
            posterior_rollout_final_only = True
            smoothed_times = None
            smoothed_dists = None

        forwarded_result = fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=None,
            obs_values=None,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            filtered_times=filtered_times,
            filtered_dists=filtered_dists,
            smoothed_times=smoothed_times,
            smoothed_dists=smoothed_dists,
            _posterior_rollout_final_only=posterior_rollout_final_only,
            **kwargs,
        )
        downstream_register = getattr(forwarded_result, "_register_numpyro_sites", None)
        if callable(downstream_register):
            downstream_register(name)

        return result


__all__ = ["LatentPathBuilder"]
