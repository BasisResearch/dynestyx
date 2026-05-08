# Smoothers

`Smoother` computes posterior smoothing distributions \(p(x_t \mid y_{1:T})\) and adds the corresponding marginal log-likelihood factor for parameter inference, mirroring `Filter` semantics.

## Usage

```python
from dynestyx import DiscreteTimeSimulator, Smoother
from dynestyx.inference.smoother_configs import KFSmootherConfig

with DiscreteTimeSimulator(n_simulations=4):
    with Smoother(
        smoother_config=KFSmootherConfig(
            filter_source="cd_dynamax",
            record_smoothed_states_mean=True,
        )
    ):
        samples = model(
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=future_times,
        )
```

`obs_times` and `obs_values` are required together. `Smoother` consumes them,
adds a marginal log-likelihood factor, and can record deterministic sites such
as `f_smoothed_states_mean`, `f_smoothed_states_cov`, and
`f_smoothed_states_cov_diag`.

## Prediction Semantics

For this release, smoother-backed prediction is intentionally future-only:
every `predict_time` must satisfy `predict_time >= max(obs_times)`. The
downstream simulator rolls out from the final smoothed state distribution.

Prediction times inside the smoothing window currently raise a clear error
instead of silently using incorrect indexing or backend-specific missing-data
behavior.

## Support Matrix

| Model class | Config | Backend |
| --- | --- | --- |
| Discrete linear-Gaussian | `KFSmootherConfig` | `cuthbert`, `cd_dynamax` |
| Discrete nonlinear Gaussian | `EKFSmootherConfig` | `cuthbert`, `cd_dynamax` |
| Discrete nonlinear Gaussian | `UKFSmootherConfig` | `cd_dynamax` |
| Discrete non-Gaussian/nonlinear | `PFSmootherConfig` | `cuthbert` |
| Continuous-discrete linear-Gaussian | `ContinuousTimeKFSmootherConfig` | `cd_dynamax` |
| Continuous-discrete nonlinear Gaussian | `ContinuousTimeEKFSmootherConfig` | `cd_dynamax` |

::: dynestyx.inference.smoothers
    options:
      members:
        - Smoother
