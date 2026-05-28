"""Internal helpers for scoring missing observations under simulator inference."""

from __future__ import annotations

import dataclasses
from typing import Protocol

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from jax import Array
from jax.errors import TracerArrayConversionError, TracerBoolConversionError

LOG_2PI = jnp.log(2.0 * jnp.pi)


@dataclasses.dataclass(frozen=True)
class MissingObservationData:
    """Preprocessed observation array with NaN-aware masks and cached summaries."""

    obs_values: Array
    safe_obs: Array
    obs_mask: Array
    row_has_any_observed: Array
    has_missing: bool
    has_partial_missing: bool
    has_fully_missing_rows: bool


def prepare_missing_observation_data(obs_values: Array) -> MissingObservationData:
    """Replace NaNs with safe fill values and cache row-level missingness summaries."""
    obs_mask = jnp.isfinite(obs_values)
    safe_obs = jnp.where(obs_mask, obs_values, jnp.zeros_like(obs_values))
    row_has_any_observed = (
        obs_mask if obs_values.ndim <= 1 else jnp.any(obs_mask, axis=-1)
    )

    try:
        obs_mask_host = np.asarray(obs_mask)
    except TracerArrayConversionError:
        obs_mask_host = None

    if obs_values.ndim <= 1:
        has_partial_missing = False
        has_fully_missing_rows = (
            bool((~obs_mask_host).any()) if obs_mask_host is not None else True
        )
    else:
        if obs_mask_host is not None:
            row_has_any_host = obs_mask_host.any(axis=-1)
            row_has_all_host = obs_mask_host.all(axis=-1)
            row_has_any_observed = jnp.asarray(row_has_any_host)
            has_partial_missing = bool((row_has_any_host & ~row_has_all_host).any())
            has_fully_missing_rows = bool((~row_has_any_host).any())
        else:
            has_partial_missing = obs_values.shape[-1] > 1
            has_fully_missing_rows = False

    try:
        has_missing = bool(np.isnan(np.asarray(obs_values)).any())
    except TracerArrayConversionError:
        has_missing = False
    return MissingObservationData(
        obs_values=obs_values,
        safe_obs=safe_obs,
        obs_mask=obs_mask,
        row_has_any_observed=row_has_any_observed,
        has_missing=has_missing,
        has_partial_missing=has_partial_missing,
        has_fully_missing_rows=has_fully_missing_rows,
    )


def _masked_independent_log_prob(
    obs_dist: dist.Independent,
    y: Array,
    obs_mask: Array,
) -> Array:
    """Exact masked log-prob for factorized Independent(..., 1) observations."""
    if obs_dist.reinterpreted_batch_ndims != 1:
        raise NotImplementedError(
            "Partial missingness currently requires Independent(..., 1) "
            "observations with one factor per observation dimension."
        )
    per_dim_lp = obs_dist.base_dist.log_prob(y)
    if jnp.ndim(per_dim_lp) == 0:
        raise NotImplementedError(
            "Partial missingness requires factorized per-dimension log-probs."
        )
    return jnp.sum(jnp.where(obs_mask, per_dim_lp, 0.0))


def _masked_multivariate_normal_log_prob(
    obs_dist: dist.MultivariateNormal,
    y: Array,
    obs_mask: Array,
) -> Array:
    """Exact masked Gaussian log-prob via a fixed-shape marginalization formula."""
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


def _is_factorized_independent_distribution(obs_dist: dist.Distribution) -> bool:
    return isinstance(obs_dist, dist.Independent) and (
        obs_dist.reinterpreted_batch_ndims == 1
    )


class MissingObservationModel(Protocol):
    """Protocol for observation callables accepted by the missingness scorer."""

    def __call__(self, x, u, t) -> dist.Distribution: ...


@dataclasses.dataclass
class MissingObservationScorer:
    """Score only observed coordinates while preserving NaNs in outputs."""

    observation_model: MissingObservationModel
    missing_data: MissingObservationData

    def score_step(self, *, x, u, t, t_idx) -> Array:
        """Return log p(y_{obs} | x, u, t) at a single observation index."""
        y = self.missing_data.safe_obs[t_idx]
        obs_mask = self.missing_data.obs_mask[t_idx]
        row_has_any_observed = self.missing_data.row_has_any_observed[t_idx]
        obs_dist = self.observation_model(x=x, u=u, t=t)

        # Scalar-like observations only need the full-row missingness rule.
        if jnp.ndim(y) == 0 or (jnp.ndim(y) == 1 and y.shape[-1] == 1):
            lp = obs_dist.log_prob(y)
            return jnp.where(row_has_any_observed, lp, jnp.zeros_like(lp))

        if isinstance(obs_dist, dist.MultivariateNormal):
            return _masked_multivariate_normal_log_prob(obs_dist, y, obs_mask)

        if _is_factorized_independent_distribution(obs_dist):
            return _masked_independent_log_prob(obs_dist, y, obs_mask)

        partial_missing = row_has_any_observed & ~jnp.all(obs_mask)
        error_msg = (
            "Partial missingness is currently supported only for "
            "marginalizable MultivariateNormal observations and "
            "factorizable Independent(..., 1) observations."
        )
        try:
            if bool(partial_missing):
                raise NotImplementedError(error_msg)
        except TracerBoolConversionError:
            y = eqx.error_if(y, partial_missing, error_msg)

        lp = obs_dist.log_prob(y)
        return jnp.where(row_has_any_observed, lp, jnp.zeros_like(lp))

    def materialize_observation(self, t_idx) -> Array:
        """Return the original NaN-preserving observation row."""
        return self.missing_data.obs_values[t_idx]


def build_missing_observation_scorer(
    *,
    observation_model: MissingObservationModel,
    missing_data: MissingObservationData,
) -> MissingObservationScorer:
    """Build a shared missingness scorer for a concrete simulator run."""
    return MissingObservationScorer(
        observation_model=observation_model,
        missing_data=missing_data,
    )
