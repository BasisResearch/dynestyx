# Filters

One of the principal functions of a dynamical systems inference engine is *filtering*, i.e., computation of the distribution \(p(x_t \mid y_{1:T}, \theta)\). In the computation of a filtering distribution, we also obtain estimates of the marginal likelihood, \(p(y_{1:T} | \theta)\), used for parameter inference/system identification. To tell `dynestyx` that a dynamical system should be processed via a filtering algorithm, we use the `Filter` class.

::: dynestyx.inference.filters
    options:
      filters: []

