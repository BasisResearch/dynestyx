# MCMC Inference

`MCMCInference` is the high-level inference wrapper for filter-based parameter inference. It wraps your model in an inference context and dispatches to the configured backend (`numpyro` or `blackjax`). `run()` keeps its samples-only return value; call `get_diagnostics()` afterward for compact sampler-specific diagnostics when supported.

For joint Gaussian random-walk proposals, use `AdaptiveMetropolisConfig`.
Its mean, full covariance, and global multiplier adapt only during warmup.
`get_diagnostics()` reports observed acceptance over retained transitions,
the final global multiplier, and the effective proposal covariance in flattened
unconstrained coordinates (including numerical diagonal regularization).
The coordinate-wise alternative is `AdaptiveMWGConfig`, whose diagnostics
retain one acceptance rate and final proposal scale per coordinate.

::: dynestyx.inference.mcmc
    options:
      members:
        - MCMCInference
