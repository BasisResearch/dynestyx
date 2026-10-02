# Evaluation

`Evaluation` consumes the `ConditionedResult` forwarded by an inner `Filter`, computes the configured evaluations, and attaches an `EvaluationResult` to it.

```python
with Evaluation(observation_scoring_config=scoring_config):
    with Filter(filter_config=filter_config):
        result = dsx.condition(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
        )

scores = result.evaluation_result.observation_scores
```

With `dsx.condition`, the scores are available on the returned result without
adding NumPyro sites. With `dsx.sample`, `Evaluation` registers the configured
scores as deterministic sites through the result's deferred callback.

::: dynestyx.evaluation.handlers
    options:
      members:
        - Evaluation
