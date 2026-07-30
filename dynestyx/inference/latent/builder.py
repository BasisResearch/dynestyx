"""NumPyro-facing explicit latent-path inference."""

from __future__ import annotations

import dataclasses
import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Bool, Int, PRNGKeyArray, Real

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.checkers import _validate_inference_supported_model_classes
from dynestyx.inference.configs.simulator import ODESimulatorConfig
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
    _stack_or_list_optional_member_values,
    _suspend_numpyro_plate_frames,
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
    assemble_completed_observations,
    prepare_missing_observation_metadata,
    resolve_missing_observation_strategy,
    validate_missing_obs_values,
)
from dynestyx.simulation.discrete import _sample_discrete_state_path
from dynestyx.simulation.utils import _sample_observation_path
from dynestyx.types import LatentStateResult
from dynestyx.utils import _build_control_path_eval

_MissingObservationMetadataCache = dict[
    tuple[str, tuple[int, ...]],
    MissingObservationMetadata,
]


def _resolve_missing_observation_metadata(
    *,
    name: str,
    dynamics: DynamicalModel,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"] | Real[Array, " obs_time"],
    missing_obs_metadata: MissingObservationMetadata | None,
    cache: _MissingObservationMetadataCache,
) -> MissingObservationMetadata:
    """Prepare static missingness layout while retaining the current times."""
    cache_key = (name, tuple(obs_values.shape))
    metadata = missing_obs_metadata
    if metadata is None:
        try:
            with jax.ensure_compile_time_eval():
                metadata = prepare_missing_observation_metadata(
                    dynamics,
                    obs_times=jnp.arange(obs_values.shape[0]),
                    obs_values=obs_values,
                )
        except ValueError as exc:
            if not isinstance(exc.__cause__, jax.errors.TracerArrayConversionError):
                raise
            metadata = cache.get(cache_key)
            if metadata is None:
                raise ValueError(
                    f"LatentPathBuilder sample site {name!r} needs a fixed "
                    "observation missingness pattern, but cannot infer one from "
                    "traced obs_values. Reuse a builder that has first seen concrete "
                    "observations, or pass metadata prepared with "
                    "dsx.prepare_missing_observation_metadata(...) to "
                    "dsx.sample(..., missing_obs_metadata=...)."
                ) from exc
    if metadata.observation_shape != tuple(obs_values.shape):
        raise ValueError(
            "missing_obs_metadata.observation_shape must match obs_values.shape."
        )
    cache[cache_key] = metadata

    time_indices = metadata.missing_flat_indices
    if len(metadata.observation_shape) > 1:
        time_indices = time_indices // metadata.observation_shape[-1]
    return dataclasses.replace(
        metadata,
        missing_obs_times=jnp.take(obs_times, time_indices, axis=0),
    )


def _sample_missing_observation_prior(
    dynamics: DynamicalModel,
    state_path: Real[Array, "state_path_time state_dim"]
    | Real[Array, " state_path_time"],
    state_path_times: Real[Array, " state_path_time"],
    obs_times: Real[Array, " obs_time"],
    ctrl_times: Real[Array, " ctrl_time"] | None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None,
    missing_flat_indices: Int[Array, " n_missing_obs"],
    key: PRNGKeyArray,
) -> Real[Array, " n_missing_obs"]:
    """Sample missing observations conditional on the reconstructed state path.

    Args:
        dynamics: Dynamical model that defines the observation distribution.
        state_path: Reconstructed state values.
        state_path_times: Times associated with `state_path`.
        obs_times: Times at which observations are required.
        ctrl_times: Times associated with `ctrl_values`.
        ctrl_values: Control values, or `None` for an uncontrolled model.
        missing_flat_indices: Positions of missing values in the flattened
            observation array.
        key: JAX pseudorandom key.

    Returns:
        Array: Sampled missing values ordered by `missing_flat_indices`.
    """
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
    dynamics: DynamicalModel,
    state_path: Real[Array, "*state_path_plate state_path_time state_dim"]
    | Real[Array, "*state_path_plate state_path_time"],
) -> list[dist.Distribution]:
    """Create one Delta distribution for each state in a reconstructed path.

    Args:
        dynamics: Dynamical model that defines the state event shape.
        state_path: Reconstructed state values.

    Returns:
        list[dist.Distribution]: Delta distributions used for posterior rollout.
    """
    event_dim = len(dynamics.initial_condition.event_shape)
    time_major_state_path = jnp.moveaxis(jnp.asarray(state_path), -(event_dim + 1), 0)
    return [
        dist.Delta(state_t, event_dim=event_dim) for state_t in time_major_state_path
    ]


class LatentPathBuilder(ObjectInterpretation, HandlesSelf):
    """Construct and score explicit latent state paths in a NumPyro model.

    Use this handler as a context manager around `dsx.sample(...)`. The builder
    creates array-valued sample sites, reconstructs the complete state path, and
    adds the joint state-observation log density to the NumPyro model.

    Several different strategies are provided for missing data, specified by the
    `missing_observation_strategy` parameter:

    - `"marginalize"` evaluates the observation density using only the observed
      components. The observation distribution must support this calculation
      (namely, a `MultivariateNormal` or `IndependentDistribution`).
    - `"augment"` creates a `"{name}_missing_obs_values"` sample site, fills the
      missing components, and evaluates the completed observation. This method
      requires a continuous observation distribution.
    - `"auto"` uses marginalization when supported and otherwise uses
      augmentation for a continuous observation distribution.
    - `"error"` rejects partially observed vectors.

    `DiracIdentityObservation` uses a separate path. Its missing components are
    inferred through `state_path_params`, and missing data support only
    `"auto"` or `"augment"`.

    A missingness layout can determine NumPyro site shapes. The builder infers
    and caches this layout whenever observations are concrete. Before an outer
    `jax.jit` passes `obs_values` dynamically, either reuse a builder that has
    seen concrete observations or pass eagerly prepared `missing_obs_metadata`
    to `dsx.sample(...)`. JAX itself does not run the function eagerly before
    tracing.

    Attributes:
        ode_simulator_config: ODE solver and integration settings used during
            deterministic continuous-time path reconstruction.
        missing_observation_strategy: Method used to handle missing entries in
            `obs_values`.
        chunk_size: Batch size passed to `jax.lax.map` while scoring transition
            and observation terms. The default, `0`, evaluates all terms with
            one `jax.vmap`. `None` maps one term at a time. A positive integer
            evaluates batches of that size with `jax.vmap`.

    Examples:
        >>> builder = dsx.LatentPathBuilder()
        >>> with builder:
        ...     result = dsx.sample(
        ...         "f",
        ...         dynamics,
        ...         obs_times=obs_times,
        ...         obs_values=obs_values,
        ...     )
    """

    def __init__(
        self,
        ode_simulator_config: ODESimulatorConfig | None = None,
        missing_observation_strategy: MissingObservationStrategy = "auto",
        chunk_size: int | None = 0,
    ) -> None:
        """Initialize explicit latent-path inference.

        Args:
            ode_simulator_config: ODE solver and integration settings used
                during deterministic continuous-time path reconstruction.
                Defaults to `ODESimulatorConfig()` when omitted.
            missing_observation_strategy: Method used to handle missing entries
                in `obs_values`, as described in the class documentation.
            chunk_size: Batch size passed to `jax.lax.map` while scoring
                transition and observation terms. The default, `0`, evaluates
                all terms with one `jax.vmap`. `None` maps one term at a time. A
                positive integer evaluates batches of that size with
                `jax.vmap`.
        """
        if ode_simulator_config is None:
            ode_simulator_config = ODESimulatorConfig()

        self.ode_simulator_config = ode_simulator_config
        self.missing_observation_strategy = missing_observation_strategy
        self.chunk_size = chunk_size
        self._missing_observation_metadata_cache: _MissingObservationMetadataCache = {}

    def _sample_single(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times: Real[Array, " obs_time"] | None,
        obs_values: Real[Array, "obs_time observation_dim"]
        | Real[Array, " obs_time"]
        | None,
        obs_values_filled: Real[Array, "obs_time observation_dim"]
        | Real[Array, " obs_time"]
        | None,
        obs_mask: Bool[Array, "obs_time observation_dim"]
        | Bool[Array, " obs_time"]
        | None,
        missing_obs_metadata: MissingObservationMetadata | None,
        ctrl_times: Real[Array, " ctrl_time"] | None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None,
        state_path_params: Real[Array, "*state_path_param_shape"] | None,
        missing_obs_values: Real[Array, " n_missing_obs"] | Real[Array, ""] | None,
    ) -> LatentStateResult:
        """Construct and score one latent state path.

        Args:
            name: Prefix used for NumPyro site names.
            dynamics: Dynamical model to condition on observations.
            obs_times: Observation times. This argument is required.
            obs_values: Observation values. This argument is required.
            obs_values_filled: Observation values with missing entries replaced
                by shape-preserving filler values.
            obs_mask: Boolean array that marks observed entries.
            missing_obs_metadata: Optional concrete missingness layout.
            ctrl_times: Times associated with `ctrl_values`.
            ctrl_values: Control values, or `None` for an uncontrolled model.
            state_path_params: State values used to construct the path. `None`
                creates an unconditioned NumPyro sample site.
            missing_obs_values: Values used to fill missing observations when
                augmentation is active. `None` creates an unconditioned sample
                site when these values are required.

        Returns:
            LatentStateResult: Reconstructed path, joint log density, and
            metadata used for missing observations and posterior rollout.

        Raises:
            ValueError: If the model is unsupported, required observations are
                absent, the missing-observation strategy is invalid, or the
                supplied latent values have incompatible shapes or semantics.
        """
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
            dynamics=dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            missing_obs_metadata=missing_obs_metadata,
            cache=self._missing_observation_metadata_cache,
        )
        expected_obs_mask = (
            jnp.ones((obs_values.size,), dtype=bool)
            .at[metadata.missing_flat_indices]
            .set(False)
            .reshape(obs_values.shape)
        )
        obs_values_filled = eqx.error_if(
            obs_values_filled,
            jnp.any(obs_mask != expected_obs_mask),
            f"obs_values missingness for sample site {name!r} does not match "
            "the LatentPathBuilder layout.",
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
                    ode_diffeqsolve_settings=(
                        self.ode_simulator_config.diffeqsolve_settings
                    ),
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
        dynamics: DynamicalModel,
        *,
        plate_shapes: tuple[int, ...] = (),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_values_filled: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_mask: Bool[Array, "*obs_value_plate obs_time observation_dim"]
        | Bool[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_has_missing: bool | None = None,
        missing_obs_metadata: MissingObservationMetadata | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
        state_path_params: Real[Array, "*state_path_param_shape"] | None = None,
        missing_obs_values: Real[Array, "*missing_obs_shape"] | None = None,
        _dsx_sample_mode: bool = False,
        **kwargs,
    ) -> LatentStateResult:
        """Construct latent paths for one trajectory or a plate of trajectories.

        Args:
            name: Prefix used for NumPyro site names.
            dynamics: Dynamical model to condition on observations.
            plate_shapes: Leading plate dimensions for independent trajectories.
            obs_times: Observation times.
            obs_values: Observation values.
            _obs_values_filled: Internal observation values with missing entries
                replaced by shape-preserving filler values.
            _obs_mask: Internal boolean array that marks observed entries.
            _obs_has_missing: Internal flag indicating whether any observations
                are missing.
            missing_obs_metadata: Optional concrete missingness layout. One
                layout is shared by all plate members.
            ctrl_times: Times associated with `ctrl_values`.
            ctrl_values: Control values, or `None` for an uncontrolled model.
            predict_times: Future times used for posterior rollout. Each time
                must be at or after the end of the reconstructed path.
            state_path_params: Optional state-path values used to condition the
                corresponding NumPyro sample site.
            missing_obs_values: Optional values used to condition the
                missing-observation sample site when augmentation is active.
            _dsx_sample_mode: Internal flag set by `dsx.sample(...)`.
            **kwargs: Additional arguments forwarded to the next handler.

        Returns:
            LatentStateResult: Path values and metadata. Plate dimensions are
            leading dimensions in array-valued fields.

        Raises:
            ValueError: If called outside `dsx.sample(...)`, if observations or
                latent values are invalid, or if `predict_times` contains an
                unsupported in-window time.
        """
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
                missing_obs_metadata=missing_obs_metadata,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                state_path_params=state_path_params,
                missing_obs_values=missing_obs_values,
            )
        else:
            member_results: list[LatentStateResult] = []
            for plate_idx in itertools.product(*[range(size) for size in plate_shapes]):
                member_name = f"{name}_p{'_'.join(str(i) for i in plate_idx)}"
                member_obs_values = _slice_array_for_plate_member(
                    obs_values, plate_shapes, plate_idx
                )
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
                            obs_values=member_obs_values,
                            obs_values_filled=_slice_array_for_plate_member(
                                _obs_values_filled, plate_shapes, plate_idx
                            ),
                            obs_mask=_slice_array_for_plate_member(
                                _obs_mask, plate_shapes, plate_idx
                            ),
                            missing_obs_metadata=missing_obs_metadata,
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

            dense_member_values = {
                attr: _stack_optional_member_values(
                    [getattr(member, attr) for member in member_results],
                    plate_shapes,
                )
                for attr in (
                    "joint_log_prob",
                    "state_path",
                    "state_path_times",
                    "completed_obs_values",
                )
            }
            ragged_member_values = {
                attr: _stack_or_list_optional_member_values(
                    [getattr(member, attr) for member in member_results],
                    plate_shapes,
                )
                for attr in (
                    "state_path_params",
                    "state_path_param_times",
                    "state_path_param_coordinate_indices",
                    "missing_obs_values",
                    "missing_obs_times",
                    "missing_obs_coordinate_indices",
                )
            }
            state_path = dense_member_values["state_path"]
            result = LatentStateResult(
                joint_log_prob=dense_member_values["joint_log_prob"],
                state_path_params=ragged_member_values["state_path_params"],
                state_path_param_times=ragged_member_values["state_path_param_times"],
                state_path_param_coordinate_indices=ragged_member_values[
                    "state_path_param_coordinate_indices"
                ],
                state_path=state_path,
                state_path_times=dense_member_values["state_path_times"],
                missing_obs_values=ragged_member_values["missing_obs_values"],
                missing_obs_times=ragged_member_values["missing_obs_times"],
                missing_obs_coordinate_indices=ragged_member_values[
                    "missing_obs_coordinate_indices"
                ],
                completed_obs_values=dense_member_values["completed_obs_values"],
                state_dists=(
                    None
                    if state_path is None
                    else _build_state_path_distributions(dynamics, state_path)
                ),
            )

        predict_times = _validate_future_only_predict_times(
            predict_times,
            result.state_path_times,
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
            filtered_times = _final_times_for_rollout(result.state_path_times)
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
