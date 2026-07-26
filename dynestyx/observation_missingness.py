"""Prepare and score observations that may contain missing values."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from jax.errors import TracerArrayConversionError, TracerBoolConversionError
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
    """Describe the missing entries in one observation array.

    Flattened indices list missing entries by time and then by component.
    A partly missing observation contains both observed and missing components.
    A fully missing observation contains no observed components.

    Attributes:
        missing_obs_times (Array): Observation time for each missing entry, in
            flattened order.
        missing_obs_coordinate_indices (Array | None): Component index for each
            missing entry in a vector observation. This is `None` for scalar
            observations.
        missing_flat_indices (Array): Positions of missing entries after
            flattening the observation array.
        observation_shape (tuple[int, ...]): Original observation-array shape.
        has_missing (bool): Whether any observation entry is missing.
        has_partial_missing (bool): Whether any observation contains both
            observed and missing components.
        has_fully_missing_rows (bool): Whether any observation contains no
            observed components.
    """

    missing_obs_times: Array
    missing_obs_coordinate_indices: Array | None
    missing_flat_indices: Array
    observation_shape: tuple[int, ...]
    has_missing: bool
    has_partial_missing: bool
    has_fully_missing_rows: bool


def _concrete_observation_mask(
    obs_values: Array | np.ndarray,
) -> np.ndarray | None:
    """Return an observed-value mask, or `None` for traced values."""
    try:
        obs_values_np = np.asarray(obs_values)
    except TracerArrayConversionError:
        return None

    if np.issubdtype(obs_values_np.dtype, np.inexact):
        return ~np.isnan(obs_values_np)
    return np.ones(obs_values_np.shape, dtype=bool)


def validate_missing_obs_values(
    missing_obs_values: Array,
    *,
    n_missing_obs: int,
) -> Array:
    """Validate values supplied for missing observation entries.

    A single missing entry accepts either a scalar or a length-one vector.
    Multiple entries require a flat vector of the exact expected length. If
    there are no missing entries, the input must be empty.

    Args:
        missing_obs_values: Values supplied for missing observation entries.
        n_missing_obs: Expected number of missing entries.

    Returns:
        Array: Validated flat vector of length `n_missing_obs`.
    """
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
    """Build missing-observation metadata from a concrete mask.

    A one-dimensional mask represents scalar observations. A two-dimensional
    mask represents vector observations with shape
    `(time, observation_dim)`.

    Args:
        obs_times: Time for each observation row.
        obs_mask: Boolean mask with `True` at observed entries.

    Returns:
        MissingObservationMetadata: Times, indices, shape, and summary flags for
            the missing entries.
    """
    try:
        obs_mask_np = np.asarray(obs_mask, dtype=bool)
        obs_times_np = np.asarray(obs_times)
    except TracerArrayConversionError as exc:
        raise ValueError(
            "Missing-observation augmentation currently requires a concrete "
            "missingness pattern. Precompute it eagerly with "
            "prepare_missing_observation_metadata(...) and pass the result via "
            "missing_obs_metadata=...."
        ) from exc

    missing_mask_np = ~obs_mask_np
    flat_missing_indices_np = np.flatnonzero(missing_mask_np.reshape(-1))
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
        missing_obs_times_np = obs_times_np[missing_mask_np]
        coord_indices_np = None
    elif obs_mask_np.ndim == 2:
        time_grid_np = np.broadcast_to(obs_times_np[:, None], obs_mask_np.shape)
        coord_grid_np = np.broadcast_to(
            np.arange(obs_mask_np.shape[-1], dtype=np.int32)[None, :],
            obs_mask_np.shape,
        )
        missing_obs_times_np = time_grid_np[missing_mask_np]
        coord_indices_np = coord_grid_np[missing_mask_np]
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
        missing_flat_indices=jnp.asarray(flat_missing_indices_np, dtype=jnp.int32),
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
        concrete_obs_mask = _concrete_observation_mask(obs_values)
        if concrete_obs_mask is not None:
            return infer_missing_observation_metadata(
                obs_times=obs_times,
                obs_mask=jnp.asarray(concrete_obs_mask),
            )
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
    validated_values = validate_missing_obs_values(
        missing_obs_values,
        n_missing_obs=missing_obs_metadata.missing_flat_indices.shape[0],
    )
    flat_obs = jnp.reshape(jnp.asarray(obs_values_filled), (-1,))
    return (
        flat_obs.at[missing_obs_metadata.missing_flat_indices]
        .set(validated_values)
        .reshape(missing_obs_metadata.observation_shape)
    )


def _masked_multivariate_normal_log_prob(
    obs_dist: dist.MultivariateNormal,
    y: Float[Array, " observation_dim"],
    obs_mask: Bool[Array, " observation_dim"],
) -> Shaped[Array, ""]:
    """Evaluate the observed marginal of a multivariate Normal distribution.

    The masked dimensions are replaced with an identity contribution so the
    Cholesky solve keeps a fixed shape across time, while the resulting scalar
    log-prob matches the exact Gaussian marginal over the observed components.

    Args:
        obs_dist: Multivariate Normal observation distribution.
        y: One filled observation vector.
        obs_mask: Boolean vector with `True` at observed components.

    Returns:
        Array: Scalar log probability of the observed components.
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

    Returns `(obs_values_filled, obs_mask, has_missing)`. `obs_values_filled`
    preserves the original array shape but replaces missing entries with
    neutral fillers so downstream scoring can keep static shapes while
    consulting `obs_mask` to decide which entries were actually observed.
    """
    if obs_values is None:
        return None, None, False

    obs_arr = jnp.asarray(obs_values)
    concrete_obs_mask = _concrete_observation_mask(obs_values)
    if jnp.issubdtype(obs_arr.dtype, jnp.inexact):
        obs_mask = ~jnp.isnan(obs_arr)
    else:
        obs_mask = jnp.ones(obs_arr.shape, dtype=bool)
    if concrete_obs_mask is not None:
        has_missing = bool(np.any(~concrete_obs_mask))
    else:
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
    """Summarize missing entries in a two-dimensional observation mask.

    For traced callers that cannot convert these summaries to Python bools, the
    three scalar flags are `False`. The row-wise JAX array remains available
    for runtime checks.

    Args:
        obs_mask: Boolean matrix shaped `(time, observation_dim)`, with `True`
            at observed entries.

    Returns:
        tuple[Array, bool, bool, bool, int]: A boolean vector marking rows with
            at least one observed component, flags for any missing entries,
            partly missing rows, and fully missing rows, and the observation
            dimension.
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


def prepare_observation_log_prob(
    dynamics: DynamicalModel,
    obs_values: (Float[Array, " time"] | Float[Array, "time observation_dim"]),
    *,
    obs_times: Array | None = None,
    precomputed_filled_obs: (
        Float[Array, " time"] | Float[Array, "time observation_dim"] | None
    ) = None,
    precomputed_obs_mask: (
        Bool[Array, " time"] | Bool[Array, "time observation_dim"] | None
    ) = None,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    missing_obs_values: Array | None = None,
    missing_obs_metadata: MissingObservationMetadata | None = None,
) -> tuple[
    Callable[..., Shaped[Array, ""]],
    Array | None,
    Array | None,
    Array | None,
]:
    """Prepare a per-time observation scorer for missing values.

    Scalar observations may have shape `(time,)`. They are represented
    internally as `(time, 1)`.

    Args:
        dynamics: Dynamical model that defines the observation distribution.
        obs_values: Scalar or vector observations, including missing entries.
        obs_times: Times associated with `obs_values`.
        precomputed_filled_obs: Observation values with missing entries replaced
            by shape-preserving filler values.
        precomputed_obs_mask: Boolean array that marks observed entries.
        missing_observation_strategy: Method used to handle missing entries.
        missing_obs_values: Values used to complete missing observations when
            augmentation is active.
        missing_obs_metadata: Positions, times, and component indices for
            `missing_obs_values`.

    Returns:
        tuple[Callable[..., Shaped[Array, ""]], Array | None, Array | None,
            Array | None]: A per-time scorer, the completed observation array
            when augmentation is active, the missing-observation times, and the
            missing-observation component indices.
    """
    obs_values = jnp.asarray(obs_values)
    scalar_observations = obs_values.ndim == 1
    original_obs_shape = tuple(obs_values.shape)
    if scalar_observations:
        obs_values = obs_values[:, None]

    if obs_times is not None:
        obs_times = jnp.asarray(obs_times)
    if precomputed_filled_obs is not None:
        precomputed_filled_obs = jnp.asarray(precomputed_filled_obs)
        if scalar_observations and precomputed_filled_obs.ndim == 1:
            precomputed_filled_obs = precomputed_filled_obs[:, None]
    if precomputed_obs_mask is not None:
        precomputed_obs_mask = jnp.asarray(precomputed_obs_mask)
        if scalar_observations and precomputed_obs_mask.ndim == 1:
            precomputed_obs_mask = precomputed_obs_mask[:, None]
    if missing_obs_values is not None:
        missing_obs_values = jnp.asarray(missing_obs_values)
        if (
            scalar_observations
            and missing_obs_metadata is None
            and tuple(missing_obs_values.shape) == original_obs_shape
        ):
            missing_obs_values = missing_obs_values[:, None]

    if (precomputed_filled_obs is None) != (precomputed_obs_mask is None):
        raise ValueError(
            "precomputed_filled_obs and precomputed_obs_mask must be provided together."
        )

    if precomputed_filled_obs is None:
        (
            filled_obs,
            obs_mask,
            row_has_any_observed,
            has_missing,
            has_partial_missing,
            _,
            observation_dim,
        ) = prepare_observation_mask(obs_values)
    else:
        assert precomputed_obs_mask is not None
        filled_obs = precomputed_filled_obs
        obs_mask = precomputed_obs_mask
        (
            row_has_any_observed,
            has_missing,
            has_partial_missing,
            _,
            observation_dim,
        ) = summarize_observation_mask(obs_mask)

    obs_shape = tuple(obs_mask.shape)

    def _metadata_shape_matches(metadata: MissingObservationMetadata) -> bool:
        return metadata.observation_shape == obs_shape or (
            scalar_observations
            and metadata.observation_shape == original_obs_shape
            and obs_shape == (obs_mask.shape[0], 1)
        )

    if missing_obs_metadata is not None:
        if not _metadata_shape_matches(missing_obs_metadata):
            raise ValueError(
                "missing_obs_metadata.observation_shape does not match the "
                "shape of obs_values for this observation scorer."
            )
        has_missing = missing_obs_metadata.has_missing
        has_partial_missing = missing_obs_metadata.has_partial_missing

    use_augmentation, expected_event_shape = resolve_missing_observation_strategy(
        dynamics,
        observation_dim=observation_dim,
        has_missing=has_missing,
        has_partial_missing=has_partial_missing,
        requested_strategy=missing_observation_strategy,
    )

    completed_obs = None
    missing_obs_times = None
    missing_obs_coordinate_indices = None
    distribution_mode: ObservationDistributionMode | Literal["augment"]
    if use_augmentation:
        metadata = missing_obs_metadata
        if (
            metadata is None
            and missing_obs_values is not None
            and tuple(jnp.asarray(missing_obs_values).shape) == tuple(filled_obs.shape)
        ):
            completed_obs = jnp.where(obs_mask, filled_obs, missing_obs_values)
        else:
            if metadata is None:
                metadata = infer_missing_observation_metadata(
                    obs_times=(
                        jnp.arange(obs_values.shape[0])
                        if obs_times is None
                        else obs_times
                    ),
                    obs_mask=obs_mask,
                )
            if not _metadata_shape_matches(metadata):
                raise ValueError(
                    "missing_obs_metadata.observation_shape does not match the "
                    "shape of obs_values for this observation scorer."
                )
            if missing_obs_values is None:
                if metadata.missing_flat_indices.shape[0] != 0:
                    raise ValueError(
                        "missing_obs_values must be provided when explicit "
                        "missing-observation augmentation is active."
                    )
                missing_obs_values = jnp.zeros((0,), dtype=filled_obs.dtype)
            completed_obs = assemble_completed_observations(
                obs_values_filled=filled_obs,
                missing_obs_values=missing_obs_values,
                missing_obs_metadata=metadata,
            )
            completed_obs = jnp.reshape(completed_obs, filled_obs.shape)
            missing_obs_times = metadata.missing_obs_times
            missing_obs_coordinate_indices = metadata.missing_obs_coordinate_indices
        distribution_mode = "augment"
    else:
        distribution_mode, expected_event_shape = (
            probe_observation_distribution_contract(
                dynamics,
                observation_dim=observation_dim,
                has_partial_missing=has_partial_missing,
            )
        )

    def _log_prob_step(*, x, u, t, t_idx) -> Shaped[Array, ""]:
        obs_dist = dynamics.observation_model(x=x, u=u, t=t)
        if distribution_mode == "augment":
            canonical_dist = _canonicalize_observation_distribution(
                obs_dist,
                observation_dim=observation_dim,
            )
            if (
                canonical_dist.is_discrete
                or tuple(canonical_dist.event_shape) != expected_event_shape
            ):
                raise ValueError(
                    "Explicit missing-observation augmentation requires the "
                    "runtime observation distribution to remain continuous "
                    "with a fixed event shape across time."
                )
            assert completed_obs is not None
            return canonical_dist.log_prob(completed_obs[t_idx])

        return masked_observation_log_prob(
            obs_dist,
            y=filled_obs[t_idx],
            obs_mask=obs_mask[t_idx],
            row_has_any_observed=row_has_any_observed[t_idx],
            observation_dim=observation_dim,
            has_partial_missing=has_partial_missing,
            expected_mode=distribution_mode,
            expected_event_shape=expected_event_shape,
        )

    returned_completed_obs = (
        completed_obs[:, 0]
        if scalar_observations and completed_obs is not None
        else completed_obs
    )
    returned_coordinate_indices = (
        None if scalar_observations else missing_obs_coordinate_indices
    )
    return (
        _log_prob_step,
        returned_completed_obs,
        missing_obs_times,
        returned_coordinate_indices,
    )
