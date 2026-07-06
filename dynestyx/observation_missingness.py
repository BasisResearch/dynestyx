"""Helpers for observation conditioning with missing data.

These utilities support both simulator conditioning and inference backends
when `obs_values` may contain NaNs. In that case we cannot always rely on
`numpyro.sample(..., obs=...)` directly, because some observation dimensions
or full rows may be missing. Instead, downstream code can evaluate only the
observed part of each likelihood term while preserving fixed array shapes.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from jax.errors import TracerBoolConversionError
from jaxtyping import Array, Bool, Float, Real, Shaped

from dynestyx.models.checkers import (
    _is_categorical_distribution,
    _make_probe_state,
    _unwrap_base_distribution,
)
from dynestyx.models.core import DynamicalModel
from dynestyx.utils import _raise_now_or_error_if

CATEGORICAL_MISSING_SENTINEL = -1
LOG_2PI = jnp.log(2.0 * jnp.pi)
ObservationDistributionMode = Literal["masked", "multivariate_normal", "independent"]
MissingObservationStrategy = Literal["auto", "marginalize", "augment", "error"]


@dataclasses.dataclass
class MissingObservationMetadata:
    """Concrete indexing metadata for explicit missing-observation augmentation.

    This metadata defines a flat latent vector ``missing_obs_values`` that
    stores only the missing entries of a dense observation array. Filling those
    values back into the missing locations reconstructs ``completed_obs_values``.
    """

    missing_obs_times: Array
    missing_obs_coordinate_indices: Array | None
    free_flat_indices: Array
    observation_shape: tuple[int, ...]
    has_missing: bool
    has_partial_missing: bool
    has_fully_missing_rows: bool


def canonicalize_missing_obs_values(
    missing_obs_values: Array,
    *,
    n_missing_obs: int,
) -> Array:
    """Canonicalize explicit missing-observation values to a flat vector."""
    values = jnp.asarray(missing_obs_values)

    if n_missing_obs == 0:
        if values.size != 0:
            raise ValueError(
                "This observation array has no missing entries. Provide an empty "
                "missing_obs_values vector."
            )
        return jnp.reshape(values, (0,))

    if values.ndim == 0 and n_missing_obs == 1:
        return jnp.reshape(values, (1,))

    if values.ndim != 1 or values.shape[0] != n_missing_obs:
        raise ValueError(
            "missing_obs_values must be a flat vector whose length matches the "
            "number of missing observation coordinates."
        )
    return values


def infer_missing_observation_metadata(
    *,
    obs_times: Array,
    obs_mask: Array,
) -> MissingObservationMetadata:
    """Infer explicit missing-observation indexing from a concrete mask."""
    try:
        obs_mask_np = np.asarray(obs_mask, dtype=bool)
        obs_times_np = np.asarray(obs_times)
    except Exception as exc:  # pragma: no cover - defensive for traced callers
        raise ValueError(
            "Missing-observation augmentation currently requires a concrete "
            "missingness pattern. Precompute it eagerly with "
            "prepare_missing_observation_metadata(...) and pass the result via "
            "missing_obs_metadata=...."
        ) from exc

    free_mask_np = ~obs_mask_np
    flat_free_indices_np = np.flatnonzero(free_mask_np.reshape(-1))
    if obs_mask_np.ndim == 1:
        row_has_any_observed_np = obs_mask_np
        row_has_all_observed_np = obs_mask_np
    else:
        row_has_any_observed_np = np.any(obs_mask_np, axis=-1)
        row_has_all_observed_np = np.all(obs_mask_np, axis=-1)
    has_missing = bool(np.any(~obs_mask_np))
    has_partial_missing = bool(
        np.any(row_has_any_observed_np & ~row_has_all_observed_np)
    )
    has_fully_missing_rows = bool(np.any(~row_has_any_observed_np))

    if obs_mask_np.ndim == 1:
        missing_obs_times_np = obs_times_np[free_mask_np]
        coord_indices_np = None
    elif obs_mask_np.ndim == 2:
        time_grid_np = np.broadcast_to(obs_times_np[:, None], obs_mask_np.shape)
        coord_grid_np = np.broadcast_to(
            np.arange(obs_mask_np.shape[-1], dtype=np.int32)[None, :],
            obs_mask_np.shape,
        )
        missing_obs_times_np = time_grid_np[free_mask_np]
        coord_indices_np = coord_grid_np[free_mask_np]
    else:
        raise ValueError(
            "Missing-observation augmentation expects obs_mask with shape (time,) "
            "or (time, observation_dim)."
        )

    obs_times_arr = jnp.asarray(obs_times)
    return MissingObservationMetadata(
        missing_obs_times=jnp.asarray(missing_obs_times_np, dtype=obs_times_arr.dtype),
        missing_obs_coordinate_indices=(
            None
            if coord_indices_np is None
            else jnp.asarray(coord_indices_np, dtype=jnp.int32)
        ),
        free_flat_indices=jnp.asarray(flat_free_indices_np, dtype=jnp.int32),
        observation_shape=tuple(obs_mask_np.shape),
        has_missing=has_missing,
        has_partial_missing=has_partial_missing,
        has_fully_missing_rows=has_fully_missing_rows,
    )


def prepare_missing_observation_metadata(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_values: Array | None = None,
    obs_mask: Array | None = None,
) -> MissingObservationMetadata:
    """Precompute augmentation metadata outside traced NumPyro/JIT contexts.

    The latent dimensionality of explicit missing-observation augmentation is
    determined by the concrete missingness pattern. Callers using traced NumPyro
    inference should therefore prepare this metadata eagerly from concrete data.
    """
    if (obs_values is None) == (obs_mask is None):
        raise ValueError(
            "Provide exactly one of obs_values or obs_mask when preparing "
            "missing-observation augmentation metadata."
        )

    if obs_mask is None:
        assert obs_values is not None
        _, obs_mask, _ = prepare_observation_views(dynamics, obs_values)
        if obs_mask is None:
            raise ValueError(
                "Could not prepare an observation mask for missing-observation "
                "augmentation metadata."
            )

    return infer_missing_observation_metadata(obs_times=obs_times, obs_mask=obs_mask)


def assemble_completed_observations(
    *,
    obs_values_filled: Array,
    missing_obs_values: Array,
    missing_obs_metadata: MissingObservationMetadata,
) -> Array:
    """Fill explicit missing-observation values into a dense observation array."""
    canonical_values = canonicalize_missing_obs_values(
        missing_obs_values,
        n_missing_obs=missing_obs_metadata.free_flat_indices.shape[0],
    )
    flat_obs = jnp.reshape(jnp.asarray(obs_values_filled), (-1,))
    return (
        flat_obs.at[missing_obs_metadata.free_flat_indices]
        .set(canonical_values)
        .reshape(missing_obs_metadata.observation_shape)
    )


def _masked_multivariate_normal_log_prob(
    obs_dist: dist.MultivariateNormal,
    y: Float[Array, " observation_dim"],
    obs_mask: Bool[Array, " observation_dim"],
) -> Shaped[Array, ""]:
    """Evaluate a masked multivariate Normal log-prob without changing array shape.

    The masked dimensions are replaced with an identity contribution so the
    Cholesky solve keeps a fixed shape across time, while the resulting scalar
    log-prob matches the exact Gaussian marginal over the observed components.
    """
    mask_f = obs_mask.astype(obs_dist.loc.dtype)
    residual = (y - obs_dist.loc) * mask_f
    cov = obs_dist.covariance_matrix
    mask_outer = mask_f[:, None] * mask_f[None, :]
    masked_cov = cov * mask_outer + jnp.diag(1.0 - mask_f)

    chol = jnp.linalg.cholesky(masked_cov)
    whitened = jsp.linalg.solve_triangular(chol, residual, lower=True)
    quad = jnp.dot(whitened, whitened)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    n_obs = jnp.sum(mask_f)
    return -0.5 * (quad + logdet + n_obs * LOG_2PI)


def _lift_scalar_observation_distribution(
    obs_dist: dist.Distribution,
) -> dist.Distribution:
    """Lift a scalar-event observation distribution to a length-1 event."""
    if obs_dist.batch_shape == ():
        return obs_dist.expand((1,)).to_event(1)
    if obs_dist.batch_shape == (1,):
        return obs_dist.to_event(1)
    raise NotImplementedError(
        "Scalar observation distributions for missingness-aware simulator "
        "conditioning must have batch shape () or (1,)."
    )


def _probe_observation_distribution(dynamics: DynamicalModel) -> dist.Distribution:
    """Probe the observation model once at a representative state."""
    x_probe = _make_probe_state(
        initial_condition=dynamics.initial_condition,
        state_dim=dynamics.state_dim,
    )
    u_probe = None if dynamics.control_dim == 0 else jnp.zeros((dynamics.control_dim,))
    t_probe = jnp.array(0.0) if dynamics.t0 is None else dynamics.t0
    return dynamics.observation_model(x=x_probe, u=u_probe, t=t_probe)


def _categorical_support_size(obs_dist: dist.Distribution) -> int:
    """Return the number of categorical labels implied by a probed distribution."""
    base = _unwrap_base_distribution(obs_dist)
    probs_or_logits = base.probs if hasattr(base, "probs") else base.logits
    return int(probs_or_logits.shape[-1])


def prepare_observation_views(
    dynamics: DynamicalModel,
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None,
) -> tuple[
    Array | None,
    Bool[Array, "*obs_value_plate obs_time observation_dim"]
    | Bool[Array, "*obs_value_plate obs_time"]
    | None,
    bool | None,
]:
    """Return mask-aware observation views for downstream scoring.

    Returns ``(obs_values_filled, obs_mask, has_missing)``. ``obs_values_filled``
    preserves the original array shape but replaces missing entries with
    neutral fillers so downstream scoring can keep static shapes while
    consulting ``obs_mask`` to decide which entries were actually observed.
    """
    if obs_values is None:
        return None, None, False

    obs_arr = jnp.asarray(obs_values)
    if jnp.issubdtype(obs_arr.dtype, jnp.inexact):
        obs_mask = ~jnp.isnan(obs_arr)
    else:
        obs_mask = jnp.ones(obs_arr.shape, dtype=bool)
    try:
        has_missing = bool(jnp.any(~obs_mask))
    except TracerBoolConversionError:
        has_missing = None

    obs_dist = _probe_observation_distribution(dynamics)
    if not _is_categorical_distribution(obs_dist):
        obs_values_filled = jnp.where(obs_mask, obs_arr, jnp.zeros_like(obs_arr))
        return obs_values_filled, obs_mask, has_missing

    def _raise_categorical_validation_error(
        invalid_mask,
        concrete_message,
        traced_message,
    ) -> None:
        try:
            has_invalid = bool(jnp.any(invalid_mask))
        except TracerBoolConversionError:
            _raise_now_or_error_if(obs_arr, jnp.any(invalid_mask), traced_message)
            return

        if not has_invalid:
            return

        bad = obs_arr[invalid_mask][0]
        raise ValueError(concrete_message(bad))

    _raise_categorical_validation_error(
        obs_mask & ~jnp.equal(obs_arr, jnp.round(obs_arr)),
        lambda bad: (
            "Categorical observations must be encoded as zero-based integer "
            f"labels. Found non-integer observed value {bad!r}."
        ),
        "Categorical observations must be encoded as zero-based integer labels.",
    )
    _raise_categorical_validation_error(
        obs_mask & (obs_arr < 0),
        lambda bad: (
            "Categorical observations must be encoded as zero-based integer "
            f"labels 0..K-1; found negative observed value {bad!r}."
        ),
        "Categorical observations must be encoded as zero-based integer labels 0..K-1.",
    )

    support_size = _categorical_support_size(obs_dist)
    _raise_categorical_validation_error(
        obs_mask & (obs_arr >= support_size),
        lambda bad: (
            "Categorical observations must be encoded as zero-based integer "
            f"labels 0..K-1 for the probed observation distribution. Found "
            f"observed value {bad!r} with K={support_size}."
        ),
        "Categorical observations must be encoded as zero-based integer labels "
        f"0..K-1 for the probed observation distribution with K={support_size}.",
    )

    obs_values_filled = jnp.where(
        obs_mask,
        obs_arr,
        jnp.asarray(CATEGORICAL_MISSING_SENTINEL, dtype=obs_arr.dtype),
    )
    return obs_values_filled.astype(jnp.int32), obs_mask, has_missing


def prepare_observation_mask(
    obs_values: Float[Array, "time observation_dim"],
) -> tuple[
    Float[Array, "time observation_dim"],
    Bool[Array, "time observation_dim"],
    Bool[Array, " time"],
    bool,
    bool,
    bool,
    int,
]:
    """Precompute row-wise missing-observation metadata from an observation array."""
    if obs_values.ndim != 2:
        raise ValueError(
            "Observation missingness expects obs_values with shape "
            "(time, observation_dim)."
        )

    obs_mask = ~jnp.isnan(obs_values)
    filled_obs = jnp.where(obs_mask, obs_values, jnp.zeros_like(obs_values))
    (
        row_has_any_observed,
        has_missing,
        has_partial_missing,
        has_fully_missing_rows,
        observation_dim,
    ) = summarize_observation_mask(obs_mask)

    return (
        filled_obs,
        obs_mask,
        row_has_any_observed,
        has_missing,
        has_partial_missing,
        has_fully_missing_rows,
        observation_dim,
    )


def summarize_observation_mask(
    obs_mask: Bool[Array, "time observation_dim"],
) -> tuple[
    Bool[Array, " time"],
    bool,
    bool,
    bool,
    int,
]:
    """Summarize row-wise missing-observation metadata from a boolean mask.

    Returns:
        row_has_any_observed: Boolean vector of shape ``(time,)`` marking rows
            with at least one observed coordinate.
        has_missing: True when any entry of ``obs_mask`` is False.
        has_partial_missing: True when at least one row mixes observed and
            missing coordinates.
        has_fully_missing_rows: True when at least one row has no observed
            coordinates at all.
        observation_dim: Size of the trailing observation-event dimension.

    For traced callers that cannot convert these summaries to Python bools, the
    three scalar flags fall back to ``False`` while the row-wise
    tensor summary remains available for downstream runtime checks.
    """
    if obs_mask.ndim != 2:
        raise ValueError(
            "Observation missingness expects obs_mask with shape "
            "(time, observation_dim)."
        )

    observation_dim = obs_mask.shape[-1]
    row_has_any_observed = jnp.any(obs_mask, axis=-1)
    row_has_all_observed = jnp.all(obs_mask, axis=-1)

    try:
        has_partial_missing = bool(
            jnp.any(row_has_any_observed & ~row_has_all_observed)
        )
        has_fully_missing_rows = bool(jnp.any(~row_has_any_observed))
        has_missing = bool(jnp.any(~obs_mask))
    except TracerBoolConversionError:
        # Traced callers still carry the row-wise boolean mask tensors. Keep the
        # summary booleans conservative here and let per-step scoring raise if an
        # unsupported masked-mode observation row turns out to be partially missing.
        has_partial_missing = False
        has_fully_missing_rows = False
        has_missing = False

    return (
        row_has_any_observed,
        has_missing,
        has_partial_missing,
        has_fully_missing_rows,
        observation_dim,
    )


def _canonicalize_observation_distribution(
    obs_dist: dist.Distribution,
    *,
    observation_dim: int,
) -> dist.Distribution:
    """Match runtime observation distributions to the row-oriented data contract."""
    if tuple(obs_dist.event_shape) != ():
        return obs_dist
    if observation_dim != 1:
        raise ValueError(
            "Scalar observation distributions are only compatible with "
            "obs_values shaped (time, 1)."
        )
    return _lift_scalar_observation_distribution(obs_dist)


def _distribution_mode(
    obs_dist: dist.Distribution,
    *,
    has_partial_missing: bool,
) -> ObservationDistributionMode:
    if isinstance(obs_dist, dist.MultivariateNormal):
        return "multivariate_normal"

    if isinstance(obs_dist, dist.Independent) and (
        obs_dist.reinterpreted_batch_ndims == 1
    ):
        return "independent"

    if has_partial_missing:
        raise NotImplementedError(
            "Partial missingness currently requires marginalizable "
            "MultivariateNormal observations or factorizable "
            "Independent(..., 1) observations."
        )

    return "masked"


def _marginalizable_distribution_mode(
    obs_dist: dist.Distribution,
) -> ObservationDistributionMode | None:
    """Return the specialized masked-likelihood mode when one is available."""
    if isinstance(obs_dist, dist.MultivariateNormal):
        return "multivariate_normal"

    if isinstance(obs_dist, dist.Independent) and (
        obs_dist.reinterpreted_batch_ndims == 1
    ):
        return "independent"

    return None


def _supports_missing_observation_augmentation(
    obs_dist: dist.Distribution,
) -> bool:
    """Return whether explicit missing-observation augmentation is supported."""
    return not obs_dist.is_discrete


def resolve_missing_observation_strategy(
    dynamics: DynamicalModel,
    *,
    observation_dim: int,
    has_missing: bool,
    has_partial_missing: bool,
    requested_strategy: MissingObservationStrategy,
) -> tuple[bool, tuple[int, ...]]:
    """Resolve whether explicit missing-observation augmentation should be used."""
    if requested_strategy not in ("auto", "marginalize", "augment", "error"):
        raise ValueError(
            "missing_observation_strategy must be one of "
            "'auto', 'marginalize', 'augment', or 'error'."
        )

    probed_obs_dist = _canonicalize_observation_distribution(
        _probe_observation_distribution(dynamics),
        observation_dim=observation_dim,
    )
    expected_event_shape = tuple(probed_obs_dist.event_shape)
    marginal_mode = _marginalizable_distribution_mode(probed_obs_dist)
    augmentation_supported = _supports_missing_observation_augmentation(probed_obs_dist)

    if requested_strategy == "error":
        if has_partial_missing:
            raise NotImplementedError(
                "Partial missingness is disabled by "
                "missing_observation_strategy='error'."
            )
        return False, expected_event_shape

    if requested_strategy == "augment":
        if has_missing and not augmentation_supported:
            raise NotImplementedError(
                "Explicit missing-observation augmentation currently requires "
                "a continuous observation family."
            )
        return has_missing, expected_event_shape

    if requested_strategy == "marginalize":
        if has_partial_missing and marginal_mode is None:
            raise NotImplementedError(
                "Partial missingness currently requires marginalizable "
                "MultivariateNormal observations or factorizable "
                "Independent(..., 1) observations unless explicit "
                "missing-observation augmentation is enabled."
            )
        return False, expected_event_shape

    if has_partial_missing and marginal_mode is None:
        if not augmentation_supported:
            raise NotImplementedError(
                "Partial missingness currently requires marginalizable "
                "MultivariateNormal observations or factorizable "
                "Independent(..., 1) observations, or a continuous "
                "observation family for explicit augmentation."
            )
        return True, expected_event_shape

    return False, expected_event_shape


def probe_observation_distribution_contract(
    dynamics: DynamicalModel,
    *,
    observation_dim: int,
    has_partial_missing: bool,
) -> tuple[ObservationDistributionMode, tuple[int, ...]]:
    """Probe a dynamics object's observation model and choose the masked mode once."""
    obs_dist = _canonicalize_observation_distribution(
        _probe_observation_distribution(dynamics),
        observation_dim=observation_dim,
    )
    return (
        _distribution_mode(obs_dist, has_partial_missing=has_partial_missing),
        tuple(obs_dist.event_shape),
    )


def masked_observation_log_prob(
    obs_dist: dist.Distribution,
    *,
    y: Array,
    obs_mask: Bool[Array, " observation_dim"],
    row_has_any_observed: Bool[Array, ""],
    observation_dim: int,
    has_partial_missing: bool,
    expected_mode: ObservationDistributionMode,
    expected_event_shape: tuple[int, ...],
) -> Shaped[Array, ""]:
    """Score only the observed portion of one observation row."""
    obs_dist = _canonicalize_observation_distribution(
        obs_dist, observation_dim=observation_dim
    )

    if has_partial_missing:
        try:
            actual_mode = _distribution_mode(
                obs_dist, has_partial_missing=has_partial_missing
            )
        except NotImplementedError as exc:
            raise ValueError(
                "Partial missingness requires a time-stable marginalizable "
                "observation family. The simulator was configured with "
                f"{expected_mode!r}, but encountered an unsupported "
                f"{type(obs_dist).__name__} at runtime."
            ) from exc

        actual_event_shape = tuple(obs_dist.event_shape)
        if actual_mode != expected_mode or actual_event_shape != expected_event_shape:
            raise ValueError(
                "Partial missingness requires the observation distribution "
                "family and event shape to remain fixed across time. "
                f"Expected mode {expected_mode!r} with event shape "
                f"{expected_event_shape}, but received mode "
                f"{actual_mode!r} with event shape {actual_event_shape}."
            )

    if expected_mode == "masked":
        row_is_partial = row_has_any_observed & ~jnp.all(obs_mask)
        _raise_now_or_error_if(
            y,
            row_is_partial,
            "Partial missingness currently requires marginalizable "
            "MultivariateNormal observations or factorizable "
            "Independent(..., 1) observations.",
        )
        return obs_dist.mask(row_has_any_observed).log_prob(y)

    if expected_mode == "independent":
        return obs_dist.base_dist.mask(obs_mask).to_event(1).log_prob(y)

    return _masked_multivariate_normal_log_prob(obs_dist, y, obs_mask)


@dataclasses.dataclass
class ObservationLogProb:
    """Evaluate conditioned observation log-probability contributions for simulators.

    This helper is used when simulator conditioning cannot be expressed as a
    simple `numpyro.sample(..., obs=obs_values[t])` call, typically because
    `obs_values` contains missing entries. It expects a single trajectory's
    observation array with shape `(time, observation_dim)`, preprocesses that
    array once, keeps both a NaN-preserving view and a filled scoring view,
    and then provides per-time-step scalar log-probability contributions of the
    form `log p(y_observed | x_t, u_t, t)`.

    For partially observed vector rows, the marginalization strategy is chosen
    from an initial probe distribution and then treated as a contract for the
    rest of the trajectory. In particular, changing the observation
    distribution family or event shape across time is not supported in that
    case. Scalar observation distributions are lifted to length-1 event
    distributions so the helper can use one row-oriented contract internally.

    When explicit missing-observation augmentation is active, this helper
    instead reconstructs a dense completed observation array and scores the
    complete-data observation density row by row.
    """

    dynamics: DynamicalModel
    obs_values: Float[Array, "time observation_dim"]
    obs_times: Array | None = None
    precomputed_filled_obs: Array | None = None
    precomputed_obs_mask: Bool[Array, "time observation_dim"] | None = None
    missing_observation_strategy: MissingObservationStrategy = "auto"
    missing_obs_values: Array | None = None
    missing_obs_metadata: MissingObservationMetadata | None = None
    distribution_mode: ObservationDistributionMode | Literal["augment"] = (
        dataclasses.field(init=False)
    )
    filled_obs: Float[Array, "time observation_dim"] = dataclasses.field(init=False)
    obs_mask: Bool[Array, "time observation_dim"] = dataclasses.field(init=False)
    completed_obs: Float[Array, "time observation_dim"] | None = dataclasses.field(
        init=False, default=None
    )
    row_has_any_observed: Bool[Array, " time"] = dataclasses.field(init=False)
    has_missing: bool = dataclasses.field(init=False)
    has_partial_missing: bool = dataclasses.field(init=False)
    has_fully_missing_rows: bool = dataclasses.field(init=False)
    observation_dim: int = dataclasses.field(init=False)
    expected_event_shape: tuple[int, ...] = dataclasses.field(
        init=False, default_factory=tuple
    )
    missing_obs_times: Array | None = dataclasses.field(init=False, default=None)
    missing_obs_coordinate_indices: Array | None = dataclasses.field(
        init=False, default=None
    )

    def __post_init__(self) -> None:
        """Precompute NaN-aware observation summaries once at construction time."""
        if (self.precomputed_filled_obs is None) != (self.precomputed_obs_mask is None):
            raise ValueError(
                "ObservationLogProb expects precomputed_filled_obs and "
                "precomputed_obs_mask to be provided together."
            )

        if self.precomputed_filled_obs is None:
            (
                self.filled_obs,
                self.obs_mask,
                self.row_has_any_observed,
                self.has_missing,
                self.has_partial_missing,
                self.has_fully_missing_rows,
                self.observation_dim,
            ) = prepare_observation_mask(self.obs_values)
        else:
            assert self.precomputed_obs_mask is not None
            self.filled_obs = self.precomputed_filled_obs
            self.obs_mask = self.precomputed_obs_mask
            (
                self.row_has_any_observed,
                self.has_missing,
                self.has_partial_missing,
                self.has_fully_missing_rows,
                self.observation_dim,
            ) = summarize_observation_mask(self.obs_mask)

        if self.missing_obs_metadata is not None:
            metadata_shape = self.missing_obs_metadata.observation_shape
            obs_shape = tuple(self.obs_mask.shape)
            scalar_lift_compatible = metadata_shape == (
                self.obs_mask.shape[0],
            ) and obs_shape == (self.obs_mask.shape[0], 1)
            if metadata_shape != obs_shape and not scalar_lift_compatible:
                raise ValueError(
                    "missing_obs_metadata.observation_shape does not match the "
                    "shape of obs_values for this observation scorer."
                )
            self.has_missing = self.missing_obs_metadata.has_missing
            self.has_partial_missing = self.missing_obs_metadata.has_partial_missing
            self.has_fully_missing_rows = (
                self.missing_obs_metadata.has_fully_missing_rows
            )

        use_augmentation, self.expected_event_shape = (
            resolve_missing_observation_strategy(
                self.dynamics,
                observation_dim=self.observation_dim,
                has_missing=self.has_missing,
                has_partial_missing=self.has_partial_missing,
                requested_strategy=self.missing_observation_strategy,
            )
        )

        if use_augmentation:
            metadata = self.missing_obs_metadata
            if metadata is None:
                metadata = infer_missing_observation_metadata(
                    obs_times=(
                        jnp.arange(self.obs_values.shape[0])
                        if self.obs_times is None
                        else self.obs_times
                    ),
                    obs_mask=self.obs_mask,
                )
            if metadata.observation_shape != tuple(self.obs_mask.shape):
                raise ValueError(
                    "missing_obs_metadata.observation_shape does not match the "
                    "shape of obs_values for this observation scorer."
                )
            if self.missing_obs_values is None:
                if metadata.free_flat_indices.shape[0] != 0:
                    raise ValueError(
                        "missing_obs_values must be provided when explicit "
                        "missing-observation augmentation is active."
                    )
                missing_obs_values = jnp.zeros((0,), dtype=self.filled_obs.dtype)
            else:
                missing_obs_values = self.missing_obs_values

            self.distribution_mode = "augment"
            self.missing_obs_times = metadata.missing_obs_times
            self.missing_obs_coordinate_indices = (
                metadata.missing_obs_coordinate_indices
            )
            self.completed_obs = assemble_completed_observations(
                obs_values_filled=self.filled_obs,
                missing_obs_values=missing_obs_values,
                missing_obs_metadata=metadata,
            )
            return

        self.distribution_mode, self.expected_event_shape = (
            probe_observation_distribution_contract(
                self.dynamics,
                observation_dim=self.observation_dim,
                has_partial_missing=self.has_partial_missing,
            )
        )
        self.completed_obs = None

    def log_prob_step(self, *, x, u, t, t_idx) -> Shaped[Array, ""]:
        """Return `log p(y_observed | x, u, t)` at one observation index.

        The returned value is a scalar log-probability contribution suitable for
        use in `numpyro.factor(...)`. Fully missing rows contribute zero, while
        partially missing vector rows are marginalized according to the mode
        chosen during initialization.
        """
        obs_dist = self.dynamics.observation_model(x=x, u=u, t=t)
        if self.distribution_mode == "augment":
            obs_dist = _canonicalize_observation_distribution(
                obs_dist,
                observation_dim=self.observation_dim,
            )
            actual_event_shape = tuple(obs_dist.event_shape)
            if obs_dist.is_discrete or actual_event_shape != self.expected_event_shape:
                raise ValueError(
                    "Explicit missing-observation augmentation requires the "
                    "runtime observation distribution to remain continuous "
                    "with a fixed event shape across time."
                )
            assert self.completed_obs is not None
            return obs_dist.log_prob(self.completed_obs[t_idx])

        return masked_observation_log_prob(
            obs_dist,
            y=self.filled_obs[t_idx],
            obs_mask=self.obs_mask[t_idx],
            row_has_any_observed=self.row_has_any_observed[t_idx],
            observation_dim=self.observation_dim,
            has_partial_missing=self.has_partial_missing,
            expected_mode=self.distribution_mode,
            expected_event_shape=self.expected_event_shape,
        )

    def observation_step(self, t_idx) -> Float[Array, " observation_dim"]:
        """Return the original NaN-preserving observation row for trace output."""
        return self.obs_values[t_idx]
