# MCMC Inference

`MCMCInference` is the high-level inference wrapper for filter-based parameter inference. It wraps your model in an inference context and dispatches to the configured backend (`numpyro` or `blackjax`). `run()` keeps its samples-only return value; call `get_diagnostics()` afterward for compact sampler-specific diagnostics when supported.

::: dynestyx.inference.mcmc
    options:
      members:
        - MCMCInference
