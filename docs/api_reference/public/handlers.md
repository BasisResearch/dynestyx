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

## Handler order

Nest handlers in this order, **outermost first**:

1. `Evaluation`
2. `Simulator`
3. `Filter`, `Smoother`, or `LatentPathBuilder`
4. [GaussianRelaxation](relaxations/gaussian_relaxation.md)
5. [Discretizer](discretizers/discretizer.md)
6. `plate`

```python
with dsx.Filter(filter_config):  # outermost
    with dsx.GaussianRelaxation(observation_model_cov=0.1):
        with dsx.Discretizer(discretizer):
            result = dsx.condition(
                "trajectory",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )
```

Outermost means the least-indented `with` block. Here, `Filter` surrounds
`GaussianRelaxation`, which surrounds `Discretizer`. The model is discretized,
then relaxed, then passed to the filter. Optional stages are omitted.

Omit stages you do not need. Use one handler per stage, choosing one of the
three inference handlers; nested plates may repeat. In a single `with`
statement, the leftmost context is outermost.

Model processing proceeds from the inside out: plates are handled first, then
discretization and relaxation prepare the model for inference, followed by
simulation and evaluation. This lets an outer inference handler receive the
transformed model.

Conditioning on `obs_values` requires an inference handler. Generating at
`predict_times` requires a simulator. Direct calls to `simulate` and `log_prob`
use the model passed to them; apply model transformations explicitly with
`discretize_dynamics` or `relax_dynamics` first when needed.

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
