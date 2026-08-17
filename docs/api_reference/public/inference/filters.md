# Filters

One of the principal functions of a dynamical systems inference engine is *filtering*, i.e., computation of the distribution \(p(x_t \mid y_{1:t}, \theta)\). In the computation of a filtering distribution, we also obtain estimates of the marginal likelihood, \(p(y_{1:T} | \theta)\), used for parameter inference/system identification. To tell `dynestyx` that a dynamical system should be processed via a filtering algorithm, we use the `Filter` class.

Where supported, `Filter` includes one-step-ahead predictive-observation outputs from the filter in its `ConditionedResult` via `include_predicted_observations`. 

::: dynestyx.inference.filters
    options:
      members:
        - Filter
