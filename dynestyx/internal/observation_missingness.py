"""Internal helpers for conditioned observation log-potentials under simulator inference."""

from __future__ import annotations

import dataclasses
from typing import Literal

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from jax import Array
from jax.errors import TracerArrayConversionError

from dynestyx.models.checkers import _make_probe_state
from dynestyx.models.core import DynamicalModel

LOG_2PI = jnp.log(2.0 * jnp.pi)


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
class ObservationLogPotential:
    """Evaluate conditioned observation log-potentials and preserve NaNs in outputs."""

    dynamics: DynamicalModel
    obs_values: Array
    distribution_mode: Literal[
        "uninitialized", "masked", "multivariate_normal", "independent"
    ] = dataclasses.field(init=False, default="uninitialized")
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

        x_probe = _make_probe_state(
            initial_condition=self.dynamics.initial_condition,
            state_dim=self.dynamics.state_dim,
        )
        u_probe = (
            None
            if self.dynamics.control_dim == 0
            else jnp.zeros((self.dynamics.control_dim,))
        )
        t_probe = jnp.array(0.0) if self.dynamics.t0 is None else self.dynamics.t0
        self._configure_from_distribution(
            self.dynamics.observation_model(x=x_probe, u=u_probe, t=t_probe)
        )

    def _is_scalar_like_row(self) -> bool:
        return self.obs_values.ndim <= 1 or self.obs_values.shape[-1] == 1

    def _configure_from_distribution(self, obs_dist: dist.Distribution) -> None:
        """Choose the log-potential evaluation mode once before scanning in time."""
        if self._is_scalar_like_row():
            self.distribution_mode = "masked"
            return

        if isinstance(obs_dist, dist.MultivariateNormal):
            self.distribution_mode = "multivariate_normal"
            return

        if isinstance(obs_dist, dist.Independent) and (
            obs_dist.reinterpreted_batch_ndims == 1
        ):
            self.distribution_mode = "independent"
            return

        if self.has_partial_missing:
            raise NotImplementedError(
                "Partial missingness currently requires marginalizable "
                "MultivariateNormal observations or factorizable "
                "Independent(..., 1) observations."
            )

        self.distribution_mode = "masked"

    def log_potential_step(self, *, x, u, t, t_idx) -> Array:
        """Return log p(y_{obs} | x, u, t) at a single observation index."""
        if self.distribution_mode == "uninitialized":
            raise RuntimeError(
                "ObservationLogPotential must be configured with an initial "
                "observation distribution before log_potential_step is used."
            )

        y = self.safe_obs[t_idx]
        obs_mask = self.obs_mask[t_idx]
        row_has_any_observed = self.row_has_any_observed[t_idx]
        obs_dist = self.dynamics.observation_model(x=x, u=u, t=t)

        if self.distribution_mode == "masked":
            return obs_dist.mask(row_has_any_observed).log_prob(y)

        if self.distribution_mode == "independent":
            return obs_dist.base_dist.mask(obs_mask).to_event(1).log_prob(y)

        return _masked_multivariate_normal_log_prob(obs_dist, y, obs_mask)

    def observation_step(self, t_idx) -> Array:
        """Return the original NaN-preserving observation row."""
        return self.obs_values[t_idx]
