# Filters

One of the principal functions of a dynamical systems inference engine is *filtering*, i.e., computation of the distribution \(p(x_t \mid y_{1:t}, \theta)\). In the computation of a filtering distribution, we also obtain estimates of the marginal likelihood, \(p(y_{1:T} | \theta)\), used for parameter inference/system identification. To tell `dynestyx` that a dynamical system should be processed via a filtering algorithm, we use the `Filter` class.

Where supported, `Filter` includes canonical one-step-ahead predictive-observation outputs in its `ConditionedResult`. `include_predicted_observations` controls that result payload, while the `record_predicted_observations_*` fields independently control NumPyro deterministic sites. Observation scoring is performed by an outer [`Evaluation`](../evaluation/handlers.md) handler rather than by `Filter` itself.

::: dynestyx.inference.filters
    options:
      members:
        - Filter
