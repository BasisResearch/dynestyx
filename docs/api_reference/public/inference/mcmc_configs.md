# MCMC Configurations

`FilterBasedMCMC` is configured via MCMC config dataclasses. These specify sampler family,
backend source, and algorithm hyperparameters.

::: dynestyx.inference.mcmc_configs
    options:
      members:
        - BaseMCMCConfig
        - NUTSConfig
        - HMCConfig
        - SGLDConfig
