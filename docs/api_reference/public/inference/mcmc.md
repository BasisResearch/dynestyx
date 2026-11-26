# MCMC Inference

`MCMCInference` runs a NumPyro model with the configured backend (`numpyro` or
`blackjax`). `run(rng_key, *args, **kwargs)` accepts positional and keyword
arguments for your model. Use `run(rng_key)` for a model that takes no arguments.
Call `get_diagnostics()` afterward for compact sampler-specific diagnostics
when supported.

For a model that calls `dsx.sample(...)`, provide observation and control data
according to your model's signature.

When using the NumPyro backend, `init_params` and `extra_fields` retain their
meaning as NumPyro `MCMC.run` options. Both backends use NumPyro helpers that can have their own keyword name collisions.

See [MCMC configurations](../configurations/mcmc_configs.md).

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
