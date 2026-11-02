# Result Types

The standalone and handler-based APIs return dataclasses.

`dynestyx.Result` provides typed properties for conditioning, latent-path,
simulation, and evaluation outputs. `sample` and `condition` return this common
interface. Unsupported properties raise `NotImplementedError`; supported fields
can be `None` when their handler did not record a value.

Filter diagnostics are available directly as `result.log_normalizing_constant`,
`result.ensemble`, `result.predicted_ensemble`, and `result.model_inputs` when the
backend records them. `.states` retains the original backend object. For example:

```python
with dsx.Filter(filter_config):
    result = dsx.condition("trajectory", dynamics, obs_times=times, obs_values=values)

# Requires a backend that records cumulative log normalizing constants.
cumulative_loglik = result.log_normalizing_constant
```

`StructuredSimulatedResult` is an explicit conversion of a simulation result,
not a direct handler result.

::: dynestyx.types.Result

::: dynestyx.types.ConditionedResult

::: dynestyx.types.LatentStateResult

::: dynestyx.types.EvaluationResult

::: dynestyx.types.SimulatedResult

::: dynestyx.types.StructuredSimulatedResult
