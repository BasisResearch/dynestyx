# Scoring

Proper scoring rules let us compare predictive observation distributions directly.

`dynestyx.evaluation.scoring` defines the reusable score-rule objects and shared scoring-policy types. The `Filter`-specific configuration layer, including `ObservationScoringConfig`, is documented on the companion [Scoring Configs](../inference/scoring_configs.md) page. Backend-specific predictive summary extraction and score execution remain in `dynestyx.inference.observation_predictions`.

::: dynestyx.evaluation.scoring
    options:
      members:
        - BaseObservationScore
        - GaussianLogProbScore
        - DawidSebastianiScore
        - ObservationWiseCRPSScore
        - EnergyScore
