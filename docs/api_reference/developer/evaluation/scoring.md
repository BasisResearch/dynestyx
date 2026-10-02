# Scoring

Proper scoring rules let us evaluate predictive observation distributions w.r.t. data in ways beyond marginal likelihood.

`dynestyx.evaluation.scoring` defines the score objects themselves: `BaseObservationScore`, `GaussianLogProbScore`, `DawidSebastianiScore`, `ObservationWiseCRPSScore`, and `EnergyScore`. These scores consume canonical one-step-ahead predictive observation distributions produced by supported filters: the continuous-time CD-Dynamax Gaussian filters (`ContinuousTimeKFConfig`, `ContinuousTimeEKFConfig`, `ContinuousTimeUKFConfig`, and `ContinuousTimeEnKFConfig`) and the discrete-time Cuthbert `EnKFConfig`.

For the Cuthbert EnKF, prediction collection retains the incoming forecast ensemble, removes Cuthbert's leading dummy state, and projects each time-aligned state ensemble through the observation mean function. The resulting `PredictedObservationOutputs` carries the projected ensemble, its mean and sample covariance, the observation-noise covariance, and the total predictive observation covariance. Cuthbert does not provide an observation-noise-perturbed ensemble, so `sample_source="auto"` uses the projected latent ensemble plus reproducibly sampled observation noise for ensemble scores. The aligned prediction at index zero is the inflated pre-update initial-prior forecast for the first observation, with no ODE transition; downstream code must not shift the sequence again.

`ObservationScoringConfig` is documented on the companion [Scoring Configs](../inference/configs/scoring_configs.md) page.

::: dynestyx.evaluation.scoring
    options:
      members:
        - BaseObservationScore
        - GaussianLogProbScore
        - DawidSebastianiScore
        - ObservationWiseCRPSScore
        - EnergyScore
