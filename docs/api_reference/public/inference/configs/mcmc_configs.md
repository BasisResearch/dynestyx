# MCMC Configurations

`MCMCInference` is configured via MCMC config dataclasses. These specify sampler family,
backend source, and algorithm hyperparameters.

Canonical imports now live under `dynestyx.inference.configs.mcmc`:

```python
from dynestyx.inference.configs.mcmc import NUTSConfig, SGLDConfig
```

::: dynestyx.inference.configs.mcmc
    options:
      members:
        - BaseMCMCConfig
        - NUTSConfig
        - HMCConfig
        - SGLDConfig
