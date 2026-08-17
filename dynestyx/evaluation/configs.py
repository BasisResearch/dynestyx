"""Configuration for evaluating conditioned dynamical-model results."""

from __future__ import annotations

import dataclasses

from dynestyx.evaluation.scoring import (
    BaseObservationScore,
    ObservationEnsembleSampleSource,
)


@dataclasses.dataclass(frozen=True)
class ObservationScoringConfig:
    """Configure proper scoring rules for predictive observations.

    Attach this configuration to ``Evaluation``. Scores are computed from the
    observed values and one-step-ahead predictive-observation outputs carried
    by the forwarded ``ConditionedResult``.
    """

    rules: tuple[BaseObservationScore, ...] = dataclasses.field(default_factory=tuple)
    record_as_numpyro_sites: bool = True
    sample_source: ObservationEnsembleSampleSource = "auto"
    sample_seed: int = 0


__all__ = ["ObservationScoringConfig"]
