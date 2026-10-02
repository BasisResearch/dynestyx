# Scoring

Proper scoring rules let us evaluate predictive observation distributions with respect to data in ways beyond marginal likelihood.

`dynestyx.evaluation.scoring` defines the score objects themselves: `BaseObservationScore`, `GaussianLogProbScore`, `DawidSebastianiScore`, `ObservationWiseCRPSScore`, and `EnergyScore`. These scores operate on canonical one-step-ahead predictive observation distributions. Supported filters include the continuous-time CD-Dynamax Gaussian filters (`ContinuousTimeKFConfig`, `ContinuousTimeEKFConfig`, `ContinuousTimeUKFConfig`, and `ContinuousTimeEnKFConfig`) and the discrete-time Cuthbert ensemble Kalman filter (`EnKFConfig`). A continuous-time deterministic model can use the latter by nesting an ODE-flow `Discretizer` inside `Filter`.

For the Cuthbert EnKF, keep `include_predicted_observations=True` (the default). With `ObservationScoringConfig(sample_source="auto")`, moment-based rules use the predictive observation mean and covariance, while ensemble-based rules use the projected forecast ensemble with observation noise added reproducibly from `sample_seed`. See [Observation scoring with a Cuthbert EnKF](../../../deep_dives/observation_scoring_with_cuthbert_enkf.ipynb) for a complete example. `ObservationScoringConfig` is documented on the companion [Scoring Configs](../inference/configs/scoring_configs.md) page.

::: dynestyx.evaluation.scoring
    options:
      members:
        - BaseObservationScore
        - GaussianLogProbScore
        - DawidSebastianiScore
        - ObservationWiseCRPSScore
        - EnergyScore
