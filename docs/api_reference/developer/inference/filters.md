# Filters

One of the principal functions of a dynamical systems inference engine is *filtering*, i.e., computation of the distribution \(p(x_t \mid y_{1:t}, \theta)\). In the computation of a filtering distribution, we also obtain estimates of the marginal likelihood, \(p(y_{1:T} | \theta)\), used for parameter inference/system identification. To tell `dynestyx` that a dynamical system should be processed via a filtering algorithm, we use the `Filter` class.

This module also hosts the public handler entry point for predicted-observation scoring via `scoring_config`. The scoring surface is documented on the [Scoring](scoring.md) page, while backend translation of predictive summaries currently lives in `dynestyx.inference.observation_predictions`.

::: dynestyx.inference.filters
    options:
      filters: []
