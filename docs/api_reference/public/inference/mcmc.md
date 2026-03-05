# MCMC Inference

`FilterBasedMCMC` is the high-level inference wrapper for filter-based parameter inference.
It wraps your model in a `Filter(...)` context and dispatches to the configured backend
(`numpyro` or `blackjax`).

::: dynestyx.inference.mcmc
    options:
      members:
        - FilterBasedMCMC
