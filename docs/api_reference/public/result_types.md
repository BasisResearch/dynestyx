# Result Types

The standalone and handler-based APIs return dataclasses.

`dynestyx.Result` is the common base class of `ConditionedResult`,
`LatentStateResult`, `SimulatedResult`, and `EvaluationResult`. The handler-based
`sample` and `condition` entry points return `Result | None`, depending on the active
handler stack. Narrow a result with `isinstance` to access fields specific to
its concrete type.

`StructuredSimulatedResult` is an explicit conversion of a simulation result,
not a direct handler result.

::: dynestyx.types.Result

::: dynestyx.types.ConditionedResult

::: dynestyx.types.LatentStateResult

::: dynestyx.types.EvaluationResult

::: dynestyx.types.SimulatedResult

::: dynestyx.types.StructuredSimulatedResult
