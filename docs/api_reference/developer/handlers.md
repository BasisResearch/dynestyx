# Handlers

The private `dynestyx.handlers._dynestyx_stack_kind()` operation reports built-in
interpretation kinds from innermost to outermost. Its default is `[]`; each
interpretation prepends its `_DynestyxStackKind` member to `fwd()`. For example,
`with Filter(), Discretizer():` reports `[DISCRETIZER, FILTER]`. Repeated kinds
are preserved, and unrelated effects do not contribute entries. This is an
internal operation, not part of the package's public API.

`condition` checks this stack before dispatch. Execution proceeds from plates
through discretization, Gaussian relaxation, inference, simulation, and evaluation. Plates may repeat;
other stages may not. Filter, Smoother, and LatentPathBuilder share one inference
stage. Observation inputs require inference, and prediction times require a
simulator, including DiscreteControlLoopSimulator. Inference without observations
and simulation without prediction times emit warnings. Existing model/backend
compatibility checks still apply.

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
