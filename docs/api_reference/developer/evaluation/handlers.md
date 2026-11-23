# Evaluation

`Evaluation` is an outer effectful handler. It consumes the `filtered_result` forwarded by `Filter`, delegates pure computation to `dynestyx.evaluation.observation_scoring`, attaches the resulting `EvaluationResult`, and returns its deferred registration callback through the handler stack.

See [Observation scoring configuration](../configurations/scoring_configs.md).

::: dynestyx.evaluation.handlers
