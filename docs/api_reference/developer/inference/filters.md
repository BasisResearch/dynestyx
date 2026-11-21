# Filters

One of the principal functions of a dynamical systems inference engine is *filtering*, i.e., computation of the distribution \(p(x_t \mid y_{1:t}, \theta)\). In the computation of a filtering distribution, we also obtain estimates of the marginal likelihood, \(p(y_{1:T} | \theta)\), used for parameter inference/system identification. To tell `dynestyx` that a dynamical system should be processed via a filtering algorithm, we use the `Filter` class.

## Discrete filters and ODE flow transitions

The Cuthbert EnKF supports both deterministic and jittered ODE flow
transitions with Gaussian observations. The Cuthbert EKF requires a
differentiable transition log density and raises `ValueError` for
`DeterministicStateEvolution`. For an ODE flow, use `jitter_scale > 0`
with the Cuthbert EKF.

The discrete CD-Dynamax EKF and UKF raise `NotImplementedError` for
both ODE flow transition types. Their shared adapter passes `t_now=0`
and `t_next=0`, but the ODE solve needs the actual interval endpoints.
The same restriction applies to discrete smoothers that use this adapter.

::: dynestyx.inference.filters
    options:
      filters: []
