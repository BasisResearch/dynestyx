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
import numpyro.distributions as dist
from jax.errors import TracerBoolConversionError
from jaxtyping import Array, Bool, Float, Shaped

from dynestyx.models.checkers import _make_probe_state
from dynestyx.models.core import DynamicalModel

LOG_2PI = jnp.log(2.0 * jnp.pi)
ObservationDistributionMode = Literal["masked", "multivariate_normal", "independent"]


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
    safe_obs = jnp.where(obs_mask, obs_values, jnp.zeros_like(obs_values))
    (
        row_has_any_observed,
        has_missing,
        has_partial_missing,
        has_fully_missing_rows,
        observation_dim,
    ) = summarize_observation_mask(obs_mask)

    return (
        safe_obs,
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
    """Summarize row-wise missing-observation metadata from a boolean mask."""
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
        has_partial_missing = observation_dim > 1
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


def probe_observation_distribution_contract(
    dynamics: DynamicalModel,
    *,
    observation_dim: int,
    has_partial_missing: bool,
) -> tuple[ObservationDistributionMode, tuple[int, ...]]:
    """Probe a dynamics object's observation model and choose the masked mode once."""
    x_probe = _make_probe_state(
        initial_condition=dynamics.initial_condition,
        state_dim=dynamics.state_dim,
    )
    u_probe = None if dynamics.control_dim == 0 else jnp.zeros((dynamics.control_dim,))
    t_probe = jnp.array(0.0) if dynamics.t0 is None else dynamics.t0
    obs_dist = _canonicalize_observation_distribution(
        dynamics.observation_model(x=x_probe, u=u_probe, t=t_probe),
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
    array once, keeps both a NaN-preserving view and a zero-filled safe view,
    and then provides per-time-step scalar log-probability contributions of the
    form `log p(y_observed | x_t, u_t, t)`.

    For partially observed vector rows, the marginalization strategy is chosen
    from an initial probe distribution and then treated as a contract for the
    rest of the trajectory. In particular, changing the observation
    distribution family or event shape across time is not supported in that
    case. Scalar observation distributions are lifted to length-1 event
    distributions so the helper can use one row-oriented contract internally.
    """

    dynamics: DynamicalModel
    obs_values: Float[Array, "time observation_dim"]
    precomputed_safe_obs: Array | None = None
    precomputed_obs_mask: Bool[Array, "time observation_dim"] | None = None
    distribution_mode: ObservationDistributionMode = dataclasses.field(init=False)
    safe_obs: Float[Array, "time observation_dim"] = dataclasses.field(init=False)
    obs_mask: Bool[Array, "time observation_dim"] = dataclasses.field(init=False)
    row_has_any_observed: Bool[Array, " time"] = dataclasses.field(init=False)
    has_missing: bool = dataclasses.field(init=False)
    has_partial_missing: bool = dataclasses.field(init=False)
    has_fully_missing_rows: bool = dataclasses.field(init=False)
    observation_dim: int = dataclasses.field(init=False)
    expected_event_shape: tuple[int, ...] = dataclasses.field(
        init=False, default_factory=tuple
    )

    def __post_init__(self) -> None:
        """Precompute NaN-aware observation summaries once at construction time."""
        if (self.precomputed_safe_obs is None) != (self.precomputed_obs_mask is None):
            raise ValueError(
                "ObservationLogProb expects precomputed_safe_obs and "
                "precomputed_obs_mask to be provided together."
            )

        if self.precomputed_safe_obs is None:
            (
                self.safe_obs,
                self.obs_mask,
                self.row_has_any_observed,
                self.has_missing,
                self.has_partial_missing,
                self.has_fully_missing_rows,
                self.observation_dim,
            ) = prepare_observation_mask(self.obs_values)
        else:
            assert self.precomputed_obs_mask is not None
            self.safe_obs = self.precomputed_safe_obs
            self.obs_mask = self.precomputed_obs_mask
            (
                self.row_has_any_observed,
                self.has_missing,
                self.has_partial_missing,
                self.has_fully_missing_rows,
                self.observation_dim,
            ) = summarize_observation_mask(self.obs_mask)
        self.distribution_mode, self.expected_event_shape = (
            probe_observation_distribution_contract(
                self.dynamics,
                observation_dim=self.observation_dim,
                has_partial_missing=self.has_partial_missing,
            )
        )

    def log_prob_step(self, *, x, u, t, t_idx) -> Shaped[Array, ""]:
        """Return `log p(y_observed | x, u, t)` at one observation index.

        The returned value is a scalar log-probability contribution suitable for
        use in `numpyro.factor(...)`. Fully missing rows contribute zero, while
        partially missing vector rows are marginalized according to the mode
        chosen during initialization.
        """
        return masked_observation_log_prob(
            self.dynamics.observation_model(x=x, u=u, t=t),
            y=self.safe_obs[t_idx],
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
