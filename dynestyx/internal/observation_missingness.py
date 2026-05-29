"""Internal helpers for missing-observation log-potentials under simulator inference."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from jax import Array
from jax.errors import TracerArrayConversionError, TracerBoolConversionError

LOG_2PI = jnp.log(2.0 * jnp.pi)


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


@dataclasses.dataclass
class MissingObservationLogPotential:
    """Evaluate missing-observation log-potentials and preserve NaNs in outputs."""

    observation_model: Callable[..., dist.Distribution]
    obs_values: Array
    safe_obs: Array = dataclasses.field(init=False)
    obs_mask: Array = dataclasses.field(init=False)
    row_has_any_observed: Array = dataclasses.field(init=False)
    has_missing: bool = dataclasses.field(init=False)
    has_partial_missing: bool = dataclasses.field(init=False)
    has_fully_missing_rows: bool = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Precompute NaN-aware observation summaries once at construction time."""
        obs_mask = ~jnp.isnan(self.obs_values)
        safe_obs = jnp.where(obs_mask, self.obs_values, jnp.zeros_like(self.obs_values))
        row_has_any_observed = (
            obs_mask if self.obs_values.ndim <= 1 else jnp.any(obs_mask, axis=-1)
        )

        try:
            obs_mask_host = np.asarray(obs_mask)
        except TracerArrayConversionError:
            obs_mask_host = None

        if self.obs_values.ndim <= 1:
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
                has_partial_missing = self.obs_values.shape[-1] > 1
                has_fully_missing_rows = False

        try:
            has_missing = bool(np.isnan(np.asarray(self.obs_values)).any())
        except TracerArrayConversionError:
            has_missing = False

        self.safe_obs = safe_obs
        self.obs_mask = obs_mask
        self.row_has_any_observed = row_has_any_observed
        self.has_missing = has_missing
        self.has_partial_missing = has_partial_missing
        self.has_fully_missing_rows = has_fully_missing_rows

    def log_potential_step(self, *, x, u, t, t_idx) -> Array:
        """Return log p(y_{obs} | x, u, t) at a single observation index."""
        y = self.safe_obs[t_idx]
        obs_mask = self.obs_mask[t_idx]
        row_has_any_observed = self.row_has_any_observed[t_idx]

        def _log_potential_observed(_) -> Array:
            obs_dist = self.observation_model(x=x, u=u, t=t)

            # Scalar-like observations only need the full-row missingness rule.
            if jnp.ndim(y) == 0 or (jnp.ndim(y) == 1 and y.shape[-1] == 1):
                return obs_dist.log_prob(y)

            if isinstance(obs_dist, dist.MultivariateNormal):
                return _masked_multivariate_normal_log_prob(obs_dist, y, obs_mask)

            if isinstance(obs_dist, dist.Independent) and (
                obs_dist.reinterpreted_batch_ndims == 1
            ):
                return _masked_independent_log_prob(obs_dist, y, obs_mask)

            partial_missing = ~jnp.all(obs_mask)
            error_msg = (
                "Partial missingness is currently supported only for "
                "marginalizable MultivariateNormal observations and "
                "factorizable Independent(..., 1) observations."
            )
            try:
                if bool(partial_missing):
                    raise NotImplementedError(error_msg)
            except TracerBoolConversionError:
                _ = eqx.error_if(y, partial_missing, error_msg)

            return obs_dist.log_prob(y)

        try:
            if not bool(row_has_any_observed):
                return jnp.zeros((), dtype=y.dtype)
            return _log_potential_observed(None)
        except TracerBoolConversionError:
            return lax.cond(
                row_has_any_observed,
                _log_potential_observed,
                lambda _: jnp.zeros((), dtype=y.dtype),
                operand=None,
            )

    def observation_step(self, t_idx) -> Array:
        """Return the original NaN-preserving observation row."""
        return self.obs_values[t_idx]
