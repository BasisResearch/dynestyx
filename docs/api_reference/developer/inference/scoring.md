# Scoring

`dynestyx.inference.scoring` defines the public scoring-rule configuration objects consumed by `Filter(..., scoring_config=...)`.

From the developer perspective, this module is intentionally narrow: it defines score objects and configuration policy for the predictive observation distribution, while backend-specific predictive summary extraction remains in `dynestyx.inference.observation_predictions`.

::: dynestyx.inference.scoring
    options:
      members:
        - BaseObservationScore
        - GaussianLogProbScore
        - DawidSebastianiScore
        - ObservationWiseCRPSScore
        - EnergyScore
        - ObservationScoringConfig
