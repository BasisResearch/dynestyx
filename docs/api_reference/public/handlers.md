# Handlers

`dynestyx` is built using [`effectful`](https://github.com/BasisResearch/effectful),
which interprets primitives according to the active handler stack.

The main user-facing primitives are:

- `sample(...)`: effectful primitive used inside handler stacks
- `condition(...)`: pure result-returning conditioning entry point
- `simulate(...)`: pure-JAX forward simulation entry point
- `log_prob(...)`: pure-JAX joint trajectory scoring entry point
- `plate(...)`: hierarchical batching primitive

For **hierarchical** models with multiple trajectories, use `plate` together
with NumPyro sampling inside the plate context. For example,

```python
with Filter(EKFConfig()):
    dsx.sample("f", dynamical_model, obs_times=obs_times, obs_values=obs_values)
```

will implement the `dsx.sample` primitive using an extended Kalman filter. For more details, see the corresponding [developer API page](../developer/handlers.md).

## Handler primitives

::: dynestyx.handlers
    options:
        members:
            - condition
            - sample
            - plate

## Pure APIs

::: dynestyx.api
    options:
        members:
            - log_prob
            - simulate
