"""Observation-prediction summaries and proper scoring rules for filters."""

from __future__ import annotations

import abc
import dataclasses
import math
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpyro.distributions as dist
from jaxtyping import Array, Float

type ScalarObservationScoreArray = Float[Array, "*plate time 1"]
type ComponentObservationScoreArray = Float[Array, "*plate time observation_dim"]
type ObservationScoreArray = (
    ScalarObservationScoreArray | ComponentObservationScoreArray
)

UnsupportedScoringPolicy = Literal["raise", "skip"]
ObservationScoringTarget = Literal["data_predictive", "latent_predictive"]
ObservationEnsembleSampleSource = Literal[
    "auto",
    "backend_ensemble",
    "latent_ensemble_plus_noise",
    "gaussian_moments",
]


def _normal_cdf(x):
    return 0.5 * (1.0 + jsp.special.erf(x / jnp.sqrt(2.0)))


def _normal_pdf(x):
    return jnp.exp(-0.5 * jnp.square(x)) / jnp.sqrt(2.0 * jnp.pi)


def _sample_gaussian_predictive_ensemble(
    *,
    pred_mean: Float[Array, ...],
    pred_cov: Float[Array, ...],
    n_samples: int,
    sample_seed: int,
) -> Float[Array, "... time n_samples observation_dim"]:
    sampled = dist.MultivariateNormal(
        loc=pred_mean,
        covariance_matrix=pred_cov,
    ).sample(jr.PRNGKey(sample_seed), sample_shape=(n_samples,))
    return jnp.moveaxis(sampled, 0, -2)


@dataclasses.dataclass(frozen=True)
class BaseObservationScore(abc.ABC):
    """Base class for proper scoring rule configurations."""

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
        obs_values: Float[Array, ...],
        pred_mean: Float[Array, ...] | None = None,
        pred_cov: Float[Array, ...] | None = None,
        pred_ensemble: Float[Array, ...] | None = None,
        **kwargs,
    ) -> ObservationScoreArray:
        raise NotImplementedError()


@dataclasses.dataclass(frozen=True)
class GaussianLogProbScore(BaseObservationScore):
    """Per-time multivariate Gaussian log-probability score.

    Returns a score array of shape ``(*plate, time, 1)``.
    """

    @property
    def default_name(self) -> str:
        return "gaussian_log_prob"

    def compute(
        self,
        *,
        obs_values: Float[Array, ...],
        pred_mean: Float[Array, ...] | None = None,
        pred_cov: Float[Array, ...] | None = None,
        **kwargs,
    ) -> ScalarObservationScoreArray:
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

    Returns a score array of shape ``(*plate, time, 1)``.
    """

    @property
    def default_name(self) -> str:
        return "dawid_sebastiani"

    def compute(
        self,
        *,
        obs_values: Float[Array, ...],
        pred_mean: Float[Array, ...] | None = None,
        pred_cov: Float[Array, ...] | None = None,
        **kwargs,
    ) -> ScalarObservationScoreArray:
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

    Returns a score array of shape ``(*plate, time, observation_dim)``.
    """

    min_variance: float = 1e-12

    @property
    def default_name(self) -> str:
        return "observation_wise_crps"

    def compute(
        self,
        *,
        obs_values: Float[Array, ...],
        pred_mean: Float[Array, ...] | None = None,
        pred_cov: Float[Array, ...] | None = None,
        **kwargs,
    ) -> ComponentObservationScoreArray:
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
    predictive moments. Returns a score array of shape ``(*plate, time, 1)``.
    """

    beta: float = 1.0
    n_samples: int | None = None
    sample_seed: int = 0

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
        obs_values: Float[Array, ...],
        pred_ensemble: Float[Array, ...] | None = None,
        pred_mean: Float[Array, ...] | None = None,
        pred_cov: Float[Array, ...] | None = None,
        **kwargs,
    ) -> ScalarObservationScoreArray:
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
                sample_seed=self.sample_seed,
            )
        obs_expanded = obs_values[..., None, :]
        first_term = jnp.mean(
            jnp.linalg.norm(pred_ensemble - obs_expanded, axis=-1) ** self.beta,
            axis=-1,
        )
        pairwise = pred_ensemble[..., :, None, :] - pred_ensemble[..., None, :, :]
        second_term = 0.5 * jnp.mean(
            jnp.linalg.norm(pairwise, axis=-1) ** self.beta,
            axis=(-2, -1),
        )
        return jnp.expand_dims(first_term - second_term, axis=-1)


# Friendly alias for the common typo.
DawadiScore = DawidSebastianiScore


@dataclasses.dataclass(frozen=True)
class ObservationScoringConfig:
    """Configuration for proper scoring rules of predicted observations."""

    rules: tuple[BaseObservationScore, ...] = dataclasses.field(default_factory=tuple)
    record_as_numpyro_sites: bool = True
    unsupported: UnsupportedScoringPolicy = "raise"
    target: ObservationScoringTarget = "data_predictive"
    sample_source: ObservationEnsembleSampleSource = "auto"
    sample_seed: int = 0


__all__ = [
    "BaseObservationScore",
    "ComponentObservationScoreArray",
    "DawadiScore",
    "DawidSebastianiScore",
    "EnergyScore",
    "GaussianLogProbScore",
    "ObservationScoreArray",
    "ObservationScoringConfig",
    "ObservationEnsembleSampleSource",
    "ObservationScoringTarget",
    "ObservationWiseCRPSScore",
    "ScalarObservationScoreArray",
    "UnsupportedScoringPolicy",
]
