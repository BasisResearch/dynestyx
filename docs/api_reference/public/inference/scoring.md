# Scoring

`dynestyx.inference.scoring` defines the proper scoring-rule surface used by `Filter(..., scoring_config=...)`.

These scores operate on the one-step-ahead predictive observation distributions produced by the filter. Recording predicted observation means, covariances, or ensembles into the trace is a separate concern controlled by the filter config. At the moment, Dynestyx supports scoring only for the continuous-time CD-Dynamax Gaussian filters (`ContinuousTimeKFConfig`, `ContinuousTimeEKFConfig`, `ContinuousTimeUKFConfig`, and `ContinuousTimeEnKFConfig`).

::: dynestyx.inference.scoring
    options:
      members:
        - BaseObservationScore
        - GaussianLogProbScore
        - DawidSebastianiScore
        - ObservationWiseCRPSScore
        - EnergyScore
        - ObservationScoringConfig
