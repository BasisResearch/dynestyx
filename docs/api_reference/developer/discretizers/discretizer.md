# Discretizers

Using continuous state evolutions can be inconvenient because they entail solving an ODE or SDE between observation times. `dynestyx` provides a `Discretizer` handler with configuration-driven methods for converting a continuous-time state evolution into a discrete-time state evolution.

See [Discretizer configurations](../configurations/discretizer_configs.md).

## ODE flow transitions

`discretize_state_evolution` with `ODEFlowConfig` integrates a deterministic
ODE from `t_now` to `t_next`, holding the control fixed over the interval.
The returned state evolution exposes the numerical endpoint through
`F(x, u, t_now, t_next)`.

| `jitter_scale` | Returned state evolution | Transition distribution |
| --- | --- | --- |
| `0` (default) | `DeterministicStateEvolution` | Delta at the numerical endpoint |
| Positive | `GaussianStateEvolution` | Gaussian centered at the numerical endpoint with covariance `jitter_scale**2 * I` |

The `Discretizer` handler uses the same conversion. Code that consumes the
discretized model can dispatch on the returned state-evolution type.

::: dynestyx.discretizers
    options:
      filters: []
      show_root_heading: false
      show_root_toc_entry: false

