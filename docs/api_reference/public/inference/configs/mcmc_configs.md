# MCMC Configurations

`MCMCInference` is configured via MCMC config dataclasses. These specify sampler family, backend source, and algorithm hyperparameters. 

`AdaptiveMetropolisConfig` makes one joint Gaussian proposal in flattened
unconstrained coordinates, adapting a full covariance and a global multiplier
during warmup. It caches the current log-density estimate across rejections.
The former coordinate-wise sampler is now `AdaptiveMWGConfig`, with its original
defaults and behavior. To retain that behavior, change existing
`AdaptiveMetropolisConfig` imports and constructors to `AdaptiveMWGConfig`;
`AdaptiveMetropolisConfig` now selects the joint method and has no
`max_adaptation` option. Both methods support only the BlackJAX backend.

::: dynestyx.inference.configs.mcmc
    options:
      members:
        - BaseMCMCConfig
        - NUTSConfig
        - HMCConfig
        - AdaptiveMWGConfig
        - AdaptiveMetropolisConfig
        - SGLDConfig
        - MALAConfig
