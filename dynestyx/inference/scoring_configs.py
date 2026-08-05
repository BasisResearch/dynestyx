"""Configuration for predictive-observation scoring w.r.t. outputs from filters."""

from __future__ import annotations

import dataclasses

from dynestyx.evaluation.scoring import (
    BaseObservationScore,
    ObservationEnsembleSampleSource,
    UnsupportedScoringPolicy,
)


@dataclasses.dataclass(frozen=True)
class ObservationScoringConfig:
    """Configuration for wiring scoring rules into `Filter`.

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


__all__ = ["ObservationScoringConfig"]
