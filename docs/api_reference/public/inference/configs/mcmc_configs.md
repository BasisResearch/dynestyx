# MCMC Configurations

`MCMCInference` is configured via MCMC config dataclasses. These specify sampler family, backend source, and algorithm hyperparameters. 

::: dynestyx.inference.configs.mcmc
    options:
      members:
        - BaseMCMCConfig
        - NUTSConfig
        - HMCConfig
        - AdaptiveMetropolisConfig
        - SGLDConfig
        - MALAConfig
