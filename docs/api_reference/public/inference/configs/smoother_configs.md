# Smoother Configurations

`Smoother` is configured using explicit `*SmootherConfig` classes.

Use smoother configs instead of filter configs when entering a `Smoother`
handler. The classes inherit the familiar filtering options, plus smoother
recording fields such as `record_smoothed_states_mean`,
`record_smoothed_states_cov_diag`, `record_smoothed_particles`, and
`record_smoothed_log_weights`.

## Common Choices

```python
from dynestyx.inference.configs.smoother import (
    ContinuousTimeKFSmootherConfig,
    EnRTSSmootherConfig,
    KFSmootherConfig,
    PFSmootherConfig,
)

kf = KFSmootherConfig(filter_source="cd_dynamax")
enrts = EnRTSSmootherConfig(n_particles=100)
pf = PFSmootherConfig(filter_source="cuthbert", n_particles=1_000)
ct_kf = ContinuousTimeKFSmootherConfig()
```

`EnRTSSmootherConfig` inherits the EnKF ensemble-size, inflation, and perturbed-observation options. `PFSmootherConfig` exposes particle-smoother options: `pf_backward_sampling_method`, `pf_mcmc_n_steps`, and `pf_n_smoother_particles`. `ContinuousTimeKFSmootherConfig` exposes `cdlgssm_smoother_type` for the CD-Dynamax continuous-discrete linear Gaussian smoother variant.

::: dynestyx.inference.configs.smoother
    options:
      filters: []
