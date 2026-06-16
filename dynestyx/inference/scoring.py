"""Proper scoring rules for predicted observations produced by ``Filter``.

Scoring is configured through :class:`ObservationScoringConfig` and is
currently supported for continuous-time CD-Dynamax Gaussian filters
(``ContinuousTimeKFConfig``, ``ContinuousTimeEKFConfig``,
``ContinuousTimeUKFConfig``, and ``ContinuousTimeEnKFConfig``).
"""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpyro.distributions as dist
from jaxtyping import Array, Float

UnsupportedScoringPolicy = Literal["raise", "skip"]
ObservationEnsembleSampleSource = Literal[
    "auto",
    "backend_ensemble",
    "latent_ensemble_plus_noise",
    "gaussian_moments",
]


def _normal_cdf(x: Float[Array, ...]) -> Float[Array, ...]:
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0)))


def _normal_pdf(x: Float[Array, ...]) -> Float[Array, ...]:
    return jnp.exp(-0.5 * jnp.square(x)) / jnp.sqrt(2.0 * jnp.pi)


def _sample_gaussian_predictive_ensemble(
    *,
    pred_mean: Float[Array, "*plate time observation_dim"],
    pred_cov: Float[Array, "*plate time observation_dim observation_dim"],
    n_samples: int,
    sample_seed: int,
) -> Float[Array, "*plate time n_samples observation_dim"]:
    sampled = dist.MultivariateNormal(
        loc=pred_mean,
        covariance_matrix=pred_cov,
    ).sample(jr.PRNGKey(sample_seed), sample_shape=(n_samples,))
    return jnp.moveaxis(sampled, 0, -2)


@dataclasses.dataclass(frozen=True)
class BaseObservationScore(abc.ABC):
    """Base class for proper scoring rule configurations.

    Subclasses define a per-time score array and the site suffix used when the
    score is recorded into the NumPyro trace.
    """

    name: str | None = None

    @property
    @abc.abstractmethod
    def default_name(self) -> str:
        raise NotImplementedError()

    @property
    def site_name(self) -> str:
        return self.name if self.name is not None else self.default_name

    @abc.abstractmethod
    def compute(
        self,
        *,
        obs_values: Float[Array, "*plate time observation_dim"],
        pred_mean: Float[Array, "*plate time observation_dim"] | None = None,
        pred_cov: Float[Array, "*plate time observation_dim observation_dim"]
        | None = None,
        pred_ensemble: Float[Array, "*plate time n_members observation_dim"]
        | None = None,
        **kwargs,
    ) -> Float[Array, "*plate time 1"] | Float[Array, "*plate time observation_dim"]:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class GaussianLogProbScore(BaseObservationScore):
    """Per-time multivariate Gaussian log-probability score.

    Uses predictive Gaussian moments and returns a score array of shape
    ``(*plate, time, 1)``. Higher values are better.
    """

    @property
    def default_name(self) -> str:
        return "gaussian_log_prob"

    def compute(
        self,
        *,
        obs_values: Float[Array, "*plate time observation_dim"],
        pred_mean: Float[Array, "*plate time observation_dim"] | None = None,
        pred_cov: Float[Array, "*plate time observation_dim observation_dim"]
        | None = None,
        **kwargs,
    ) -> Float[Array, "*plate time 1"]:
        if pred_mean is None or pred_cov is None:
            raise ValueError(
                "GaussianLogProbScore requires Gaussian predictive mean and covariance."
            )
        lp = dist.MultivariateNormal(
            loc=pred_mean,
            covariance_matrix=pred_cov,
        ).log_prob(obs_values)
        return jnp.expand_dims(lp, axis=-1)


@dataclasses.dataclass(frozen=True)
class DawidSebastianiScore(BaseObservationScore):
    """Per-time Dawid-Sebastiani score under Gaussian predictive moments.

    Uses predictive Gaussian moments and returns a score array of shape
    ``(*plate, time, 1)``. Lower values are better.
    """

    @property
    def default_name(self) -> str:
        return "dawid_sebastiani"

    def compute(
        self,
        *,
        obs_values: Float[Array, "*plate time observation_dim"],
        pred_mean: Float[Array, "*plate time observation_dim"] | None = None,
        pred_cov: Float[Array, "*plate time observation_dim observation_dim"]
        | None = None,
        **kwargs,
    ) -> Float[Array, "*plate time 1"]:
        if pred_mean is None or pred_cov is None:
            raise ValueError(
                "DawidSebastianiScore requires Gaussian predictive mean and covariance."
            )
        innovation = obs_values - pred_mean
        solved = jnp.linalg.solve(pred_cov, innovation[..., None])[..., 0]
        mahal = jnp.sum(innovation * solved, axis=-1)
        _, logdet = jnp.linalg.slogdet(pred_cov)
        return jnp.expand_dims(logdet + mahal, axis=-1)


@dataclasses.dataclass(frozen=True)
class ObservationWiseCRPSScore(BaseObservationScore):
    """Per-observation-component CRPS under Gaussian predictive marginals.

    Applies the scalar Gaussian CRPS to each observation component separately
    and returns a score array of shape ``(*plate, time, observation_dim)``.
    Lower values are better.
    """

    min_variance: float = 1e-12

    @property
    def default_name(self) -> str:
        return "observation_wise_crps"

    def compute(
        self,
        *,
        obs_values: Float[Array, "*plate time observation_dim"],
        pred_mean: Float[Array, "*plate time observation_dim"] | None = None,
        pred_cov: Float[Array, "*plate time observation_dim observation_dim"]
        | None = None,
        **kwargs,
    ) -> Float[Array, "*plate time observation_dim"]:
        if pred_mean is None or pred_cov is None:
            raise ValueError(
                "ObservationWiseCRPSScore requires Gaussian predictive mean and covariance."
            )
        variances = jnp.diagonal(pred_cov, axis1=-2, axis2=-1)
        scales = jnp.sqrt(jnp.maximum(variances, self.min_variance))
        z = (obs_values - pred_mean) / scales
        return scales * (
            z * (2.0 * _normal_cdf(z) - 1.0)
            + 2.0 * _normal_pdf(z)
            - 1.0 / math.sqrt(math.pi)
        )


@dataclasses.dataclass(frozen=True)
class EnergyScore(BaseObservationScore):
    """Per-time ensemble energy score with exponent ``beta``.

    If an explicit predictive observation ensemble is unavailable, this score
    can approximate one by drawing ``n_samples`` observations from the Gaussian
    predictive observation moments. Returns a score array of shape
    ``(*plate, time, 1)``. Lower values are better.

    Under `Filter(..., scoring_config=...)`, any synthetic sampling needed by
    this score uses `ObservationScoringConfig.sample_seed`.

    `vectorized_pairwise=True` is faster for moderate ensemble sizes but
    materializes the full pairwise distance tensor. Setting it to `False`
    uses a lower-memory `lax.scan` path at the cost of extra compute.
    """

    beta: float = 1.0
    n_samples: int | None = None
    vectorized_pairwise: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.beta < 2.0):
            raise ValueError("EnergyScore requires 0 < beta < 2.")
        if self.n_samples is not None and self.n_samples <= 0:
            raise ValueError("EnergyScore requires n_samples to be positive.")

    @property
    def default_name(self) -> str:
        if self.beta == 1.0:
            return "energy_score"
        return f"energy_score_beta_{str(self.beta).replace('.', '_')}"

    def compute(
        self,
        *,
        obs_values: Float[Array, "*plate time observation_dim"],
        pred_ensemble: Float[Array, "*plate time n_members observation_dim"]
        | None = None,
        pred_mean: Float[Array, "*plate time observation_dim"] | None = None,
        pred_cov: Float[Array, "*plate time observation_dim observation_dim"]
        | None = None,
        sample_seed: int = 0,
        **kwargs,
    ) -> Float[Array, "*plate time 1"]:
        if pred_ensemble is None:
            if self.n_samples is None:
                raise NotImplementedError(
                    "EnergyScore requires either predicted observation ensembles "
                    "or Gaussian predictive moments together with `n_samples`."
                )
            if pred_mean is None or pred_cov is None:
                raise NotImplementedError(
                    "EnergyScore could not synthesize a predictive ensemble "
                    "because Gaussian predictive moments were unavailable."
                )
            pred_ensemble = _sample_gaussian_predictive_ensemble(
                pred_mean=pred_mean,
                pred_cov=pred_cov,
                n_samples=self.n_samples,
                sample_seed=sample_seed,
            )
        obs_expanded = obs_values[..., None, :]
        first_term = jnp.mean(
            jnp.linalg.norm(pred_ensemble - obs_expanded, axis=-1) ** self.beta,
            axis=-1,
        )
        if self.vectorized_pairwise:
            pairwise = pred_ensemble[..., :, None, :] - pred_ensemble[..., None, :, :]
            second_term = 0.5 * jnp.mean(
                jnp.linalg.norm(pairwise, axis=-1) ** self.beta,
                axis=(-2, -1),
            )
        else:
            n_members = pred_ensemble.shape[-2]
            members_first = jnp.moveaxis(pred_ensemble, -2, 0)

            def scan_step(
                total: Float[Array, "*plate time"],
                member: Float[Array, "*plate time observation_dim"],
            ) -> tuple[Float[Array, "*plate time"], None]:
                distances = (
                    jnp.linalg.norm(
                        pred_ensemble - member[..., None, :],
                        axis=-1,
                    )
                    ** self.beta
                )
                return total + jnp.sum(distances, axis=-1), None

            total0 = jnp.zeros(pred_ensemble.shape[:-2], dtype=pred_ensemble.dtype)
            total, _ = jax.lax.scan(scan_step, total0, members_first)
            second_term = 0.5 * total / float(n_members * n_members)
        return jnp.expand_dims(first_term - second_term, axis=-1)


@dataclasses.dataclass(frozen=True)
class ObservationScoringConfig:
    """Configuration for predicted-observation diagnostics and scoring rules.

    Attach an instance to ``Filter(..., scoring_config=...)`` to request
    per-time score arrays for the one-step-ahead predictive observation
    distributions produced by the filter.

    Scoring currently supports only continuous-time CD-Dynamax Gaussian
    filters (`ContinuousTimeKFConfig`, `ContinuousTimeEKFConfig`,
    `ContinuousTimeUKFConfig`, and `ContinuousTimeEnKFConfig`).

    Attributes:
        rules: Ordered tuple of score definitions to evaluate at each
            observation time.
        record_as_numpyro_sites: If `True`, record each computed score array
            as a `numpyro.deterministic` site named
            ``{sample_name}_{rule.site_name}``. This is independent of the
            `record_predicted_observations_*` fields on the filter config:
            score recording does not require predictive summary recording.
        unsupported: Policy for requested score rules or sampling modes that
            are unavailable for the active filter backend. `"raise"` fails
            immediately; `"skip"` silently omits unsupported rules.
        sample_source: Strategy for obtaining predictive observation
            ensembles when a rule needs samples. `"auto"` prefers a
            backend-provided predictive observation ensemble, then falls back
            to adding observation noise to a latent predictive ensemble, and
            finally to Gaussian moments if the rule supports that path.
        sample_seed: PRNG seed used whenever Dynestyx needs to synthesize
            predictive observation samples during scoring, whether by adding
            observation noise to a latent predictive ensemble or by drawing
            from Gaussian predictive moments for a rule such as `EnergyScore`.
    """

    rules: tuple[BaseObservationScore, ...] = dataclasses.field(default_factory=tuple)
    record_as_numpyro_sites: bool = True
    unsupported: UnsupportedScoringPolicy = "raise"
    sample_source: ObservationEnsembleSampleSource = "auto"
    sample_seed: int = 0


__all__ = [
    "BaseObservationScore",
    "DawidSebastianiScore",
    "EnergyScore",
    "GaussianLogProbScore",
    "ObservationScoringConfig",
    "ObservationEnsembleSampleSource",
    "ObservationWiseCRPSScore",
    "UnsupportedScoringPolicy",
]
