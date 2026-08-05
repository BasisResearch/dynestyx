# Filters

One of the principal functions of a dynamical systems inference engine is *filtering*, i.e., computation of the distribution \(p(x_t \mid y_{1:t}, \theta)\). In the computation of a filtering distribution, we also obtain estimates of the marginal likelihood, \(p(y_{1:T} | \theta)\), used for parameter inference/system identification. To tell `dynestyx` that a dynamical system should be processed via a filtering algorithm, we use the `Filter` class.

`Filter` can also expose one-step-ahead predictive observation diagnostics through `scoring_config` and the `record_predicted_observations_*` fields on the chosen filter config. Scoring always refers to the predictive observation distribution, while the recording flags only control whether backend predictive summaries are written into the trace. This path currently supports the continuous-time CD-Dynamax Gaussian filters (`ContinuousTimeKFConfig`, `ContinuousTimeEKFConfig`, `ContinuousTimeUKFConfig`, and `ContinuousTimeEnKFConfig`). See [Scoring](../evaluation/scoring.md) for score definitions and configuration.

::: dynestyx.inference.filters
    options:
      members:
        - Filter
