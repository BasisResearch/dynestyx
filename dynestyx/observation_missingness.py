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
from jax.errors import TracerArrayConversionError
from jaxtyping import Array, Bool, Float, Shaped

from dynestyx.models.checkers import _make_probe_state
from dynestyx.models.core import DynamicalModel

LOG_2PI = jnp.log(2.0 * jnp.pi)
ObservationDistributionMode = Literal[
    "uninitialized", "masked", "multivariate_normal", "independent"
]


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


def _coerce_observation_value_for_distribution(
    obs_dist: dist.Distribution,
    y: Float[Array, " observation_dim"],
):
    """Coerce safe observation values to a distribution-compatible dtype if needed."""
    if type(obs_dist).__name__.startswith("Categorical"):
        return y.astype(jnp.int32)

    if isinstance(obs_dist, dist.Independent) and (
        obs_dist.reinterpreted_batch_ndims == 1
    ):
        return _coerce_observation_value_for_distribution(obs_dist.base_dist, y)

    if hasattr(obs_dist, "base_dist"):
        return _coerce_observation_value_for_distribution(obs_dist.base_dist, y)

    return y


@dataclasses.dataclass
class ObservationMask:
    """Precompute row-wise missing-observation metadata from an observation array.

    The input must be shaped `(time, observation_dim)`, even for scalar
    observation models where `observation_dim == 1`.
    """

    obs_values: Float[Array, "time observation_dim"]
    safe_obs: Float[Array, "time observation_dim"] = dataclasses.field(init=False)
    obs_mask: Bool[Array, "time observation_dim"] = dataclasses.field(init=False)
    row_has_any_observed: Bool[Array, " time"] = dataclasses.field(init=False)
    has_missing: bool = dataclasses.field(init=False)
    has_partial_missing: bool = dataclasses.field(init=False)
    has_fully_missing_rows: bool = dataclasses.field(init=False)
    observation_dim: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if self.obs_values.ndim != 2:
            raise ValueError(
                "ObservationMask expects obs_values with shape (time, observation_dim)."
            )

        self.observation_dim = self.obs_values.shape[-1]
        obs_mask = ~jnp.isnan(self.obs_values)
        safe_obs = jnp.where(obs_mask, self.obs_values, jnp.zeros_like(self.obs_values))
        row_has_any_observed = jnp.any(obs_mask, axis=-1)

        try:
            obs_mask_host = np.asarray(obs_mask)
        except TracerArrayConversionError:
            obs_mask_host = None

        if obs_mask_host is not None:
            row_has_any_host = obs_mask_host.any(axis=-1)
            row_has_all_host = obs_mask_host.all(axis=-1)
            row_has_any_observed = jnp.asarray(row_has_any_host)
            has_partial_missing = bool((row_has_any_host & ~row_has_all_host).any())
            has_fully_missing_rows = bool((~row_has_any_host).any())
        else:
            has_partial_missing = self.observation_dim > 1
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


@dataclasses.dataclass
class ObservationDistributionContract:
    """Describe how masked log-probabilities should be evaluated over time."""

    observation_dim: int
    has_partial_missing: bool
    distribution_mode: ObservationDistributionMode = dataclasses.field(
        init=False, default="uninitialized"
    )
    expected_event_shape: tuple[int, ...] = dataclasses.field(
        init=False, default_factory=tuple
    )

    def canonicalize(self, obs_dist: dist.Distribution) -> dist.Distribution:
        """Match runtime observation distributions to the row-oriented data contract."""
        if tuple(obs_dist.event_shape) != ():
            return obs_dist
        if self.observation_dim != 1:
            raise ValueError(
                "Scalar observation distributions are only compatible with "
                "obs_values shaped (time, 1)."
            )
        return _lift_scalar_observation_distribution(obs_dist)

    def _distribution_mode(
        self,
        obs_dist: dist.Distribution,
    ) -> Literal["masked", "multivariate_normal", "independent"]:
        if isinstance(obs_dist, dist.MultivariateNormal):
            return "multivariate_normal"

        if isinstance(obs_dist, dist.Independent) and (
            obs_dist.reinterpreted_batch_ndims == 1
        ):
            return "independent"

        if self.has_partial_missing:
            raise NotImplementedError(
                "Partial missingness currently requires marginalizable "
                "MultivariateNormal observations or factorizable "
                "Independent(..., 1) observations."
            )

        return "masked"

    def configure_from_distribution(self, obs_dist: dist.Distribution) -> None:
        """Choose the masked log-probability strategy before iterating in time."""
        obs_dist = self.canonicalize(obs_dist)
        self.distribution_mode = self._distribution_mode(obs_dist)
        self.expected_event_shape = tuple(obs_dist.event_shape)

    def validate_runtime_distribution(self, obs_dist: dist.Distribution) -> None:
        """Check that partial-missingness assumptions hold at runtime."""
        if not self.has_partial_missing:
            return

        obs_dist = self.canonicalize(obs_dist)
        try:
            actual_mode = self._distribution_mode(obs_dist)
        except NotImplementedError as exc:
            raise ValueError(
                "Partial missingness requires a time-stable marginalizable "
                "observation family. The simulator was configured with "
                f"{self.distribution_mode!r}, but encountered an unsupported "
                f"{type(obs_dist).__name__} at runtime."
            ) from exc

        actual_event_shape = tuple(obs_dist.event_shape)
        if (
            actual_mode != self.distribution_mode
            or actual_event_shape != self.expected_event_shape
        ):
            raise ValueError(
                "Partial missingness requires the observation distribution "
                "family and event shape to remain fixed across time. "
                f"Expected mode {self.distribution_mode!r} with event shape "
                f"{self.expected_event_shape}, but received mode "
                f"{actual_mode!r} with event shape {actual_event_shape}."
            )


def make_observation_distribution_contract(
    dynamics: DynamicalModel,
    *,
    observation_dim: int,
    has_partial_missing: bool,
) -> ObservationDistributionContract:
    """Probe a dynamics object's observation model and build its masked contract."""
    contract = ObservationDistributionContract(
        observation_dim=observation_dim,
        has_partial_missing=has_partial_missing,
    )
    x_probe = _make_probe_state(
        initial_condition=dynamics.initial_condition,
        state_dim=dynamics.state_dim,
    )
    u_probe = None if dynamics.control_dim == 0 else jnp.zeros((dynamics.control_dim,))
    t_probe = jnp.array(0.0) if dynamics.t0 is None else dynamics.t0
    contract.configure_from_distribution(
        dynamics.observation_model(x=x_probe, u=u_probe, t=t_probe)
    )
    return contract


def masked_observation_log_prob(
    obs_dist: dist.Distribution,
    *,
    y: Float[Array, " observation_dim"],
    obs_mask: Bool[Array, " observation_dim"],
    row_has_any_observed: Bool[Array, ""],
    contract: ObservationDistributionContract,
) -> Shaped[Array, ""]:
    """Score only the observed portion of one observation row."""
    obs_dist = contract.canonicalize(obs_dist)
    contract.validate_runtime_distribution(obs_dist)
    y_scored = _coerce_observation_value_for_distribution(obs_dist, y)

    if contract.distribution_mode == "uninitialized":
        raise RuntimeError(
            "ObservationDistributionContract must be configured before "
            "masked_observation_log_prob is used."
        )

    if contract.distribution_mode == "masked":
        return obs_dist.mask(row_has_any_observed).log_prob(y_scored)

    if contract.distribution_mode == "independent":
        return obs_dist.base_dist.mask(obs_mask).to_event(1).log_prob(y_scored)

    return _masked_multivariate_normal_log_prob(obs_dist, y_scored, obs_mask)


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
    distribution_mode: ObservationDistributionMode = dataclasses.field(
        init=False, default="uninitialized"
    )
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
    contract: ObservationDistributionContract = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Precompute NaN-aware observation summaries once at construction time."""
        mask = ObservationMask(self.obs_values)
        self.observation_dim = mask.observation_dim
        self.safe_obs = mask.safe_obs
        self.obs_mask = mask.obs_mask
        self.row_has_any_observed = mask.row_has_any_observed
        self.has_missing = mask.has_missing
        self.has_partial_missing = mask.has_partial_missing
        self.has_fully_missing_rows = mask.has_fully_missing_rows
        self.contract = make_observation_distribution_contract(
            self.dynamics,
            observation_dim=self.observation_dim,
            has_partial_missing=self.has_partial_missing,
        )
        self.distribution_mode = self.contract.distribution_mode
        self.expected_event_shape = self.contract.expected_event_shape

    def _canonicalize_observation_distribution(
        self, obs_dist: dist.Distribution
    ) -> dist.Distribution:
        """Match runtime observation distributions to the row-oriented data contract."""
        return self.contract.canonicalize(obs_dist)

    def _configure_from_distribution(self, obs_dist: dist.Distribution) -> None:
        """Choose the log-probability evaluation mode once before scanning in time.

        For partial missingness, the chosen mode is a contract: later
        observation distributions must stay in the same supported family and
        preserve the same event shape.
        """
        self.contract.configure_from_distribution(obs_dist)
        self.distribution_mode = self.contract.distribution_mode
        self.expected_event_shape = self.contract.expected_event_shape

    def _distribution_mode(
        self,
        obs_dist: dist.Distribution,
    ) -> Literal["masked", "multivariate_normal", "independent"]:
        obs_dist = self._canonicalize_observation_distribution(obs_dist)
        return self.contract._distribution_mode(obs_dist)

    def _validate_partial_missing_distribution(
        self, obs_dist: dist.Distribution
    ) -> None:
        """Check the partial-missingness distribution contract at a time step."""
        self.contract.validate_runtime_distribution(obs_dist)

    def log_prob_step(self, *, x, u, t, t_idx) -> Shaped[Array, ""]:
        """Return `log p(y_observed | x, u, t)` at one observation index.

        The returned value is a scalar log-probability contribution suitable for
        use in `numpyro.factor(...)`. Fully missing rows contribute zero, while
        partially missing vector rows are marginalized according to the mode
        chosen during initialization.
        """
        if self.distribution_mode == "uninitialized":
            raise RuntimeError(
                "ObservationLogProb must be configured with an initial "
                "observation distribution before log_prob_step is used."
            )

        y = self.safe_obs[t_idx]
        obs_mask = self.obs_mask[t_idx]
        row_has_any_observed = self.row_has_any_observed[t_idx]
        obs_dist = self.dynamics.observation_model(x=x, u=u, t=t)
        return masked_observation_log_prob(
            obs_dist,
            y=y,
            obs_mask=obs_mask,
            row_has_any_observed=row_has_any_observed,
            contract=self.contract,
        )

    def observation_step(self, t_idx) -> Float[Array, " observation_dim"]:
        """Return the original NaN-preserving observation row for trace output."""
        return self.obs_values[t_idx]
