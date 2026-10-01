# Handlers

The private `dynestyx.handlers._dynestyx_stack_kind()` operation reports built-in
interpretation kinds from innermost to outermost. Its default is `[]`; each
interpretation prepends its `_DynestyxStackKind` member to `fwd()`. For example,
`with Filter(), Discretizer():` reports `[DISCRETIZER, FILTER]`. Repeated kinds
are preserved, and unrelated effects do not contribute entries. This is an
internal operation, not part of the package's public API.

`condition` checks this stack before dispatch. Execution proceeds from plates
through discretization, inference, simulation, and evaluation. Plates may repeat;
other stages may not. Filter, Smoother, and LatentPathBuilder share one inference
stage. Observation inputs require inference, and prediction times require a
simulator, including DiscreteControlLoopSimulator; missing stages are supplied
by the private default interpretation.
Inference without observations and simulation without prediction times emit
warnings. Existing model/backend compatibility checks still apply.

Default completion happens before Simulator/Evaluation consume their inputs, or
in `_condition_intp`'s default rule. `_defaults` builds ordinary interpretation
objects and takes their coproduct with the existing continuation. Applying that
composed method directly preserves its enclosing `fwd` continuation and unrelated
effects, without replacing the ambient interpretation or replaying inner handlers.
Forwarded conditioning results and `_dsx_prediction_done` prevent duplicate work.

Stack-query implementations accept `**kwargs` to establish an empty argument
frame in effectful even when queried inside an operation with arguments.

::: dynestyx.handlers
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - condition
        - sample
        - plate

::: dynestyx.api
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members:
        - log_prob
        - simulate
