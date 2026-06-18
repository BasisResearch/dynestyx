# Scoring

Proper scoring rules let us evaluate predictive observation distributions w.r.t. data in ways beyond marginal likelihood.

`dynestyx.evaluation.scoring` defines the score objects themselves: `BaseObservationScore`, `GaussianLogProbScore`, `DawidSebastianiScore`, `ObservationWiseCRPSScore`, and `EnergyScore`. These scores currently operate on the one-step-ahead predictive observation distributions produced by the continuous-time CD-Dynamax Gaussian filters (`ContinuousTimeKFConfig`, `ContinuousTimeEKFConfig`, `ContinuousTimeUKFConfig`, and `ContinuousTimeEnKFConfig`). `ObservationScoringConfig` is documented on the companion [Scoring Configs](../inference/scoring_configs.md) page.

::: dynestyx.evaluation.scoring
    options:
      members:
        - BaseObservationScore
        - GaussianLogProbScore
        - DawidSebastianiScore
        - ObservationWiseCRPSScore
        - EnergyScore
