# Smoothing

Smoothers estimate the latent state at each time using the whole observation window, both past and future. They give better state estimates than a filter when the goal is to reconstruct a trajectory after the fact rather than online. Read [reference.md](reference.md) first.

## Choosing a smoother

The smoother families mirror the filters, so pick by the same model shape.

- Linear Gaussian → `KFSmootherConfig`, the exact Rauch-Tung-Striebel smoother.
- Nonlinear, roughly Gaussian → `EKFSmootherConfig` or `UKFSmootherConfig`.
- Nonlinear and non-Gaussian → `PFSmootherConfig`, the particle smoother with backward sampling.

Continuous-time variants exist as `ContinuousTimeKFSmootherConfig` and `ContinuousTimeEKFSmootherConfig` for models with an SDE between observations. Import these from `dynestyx.inference.smoothers`. `Smoother` is exported from the top level.

## Run a smoother

The pattern matches a filter. Wrap the model in `Smoother`, request the smoothed states, and read them from the trace.

```python
import jax.random as jr
from numpyro.infer import Predictive
from dynestyx import Smoother
from dynestyx.inference.smoothers import KFSmootherConfig

with Smoother(smoother_config=KFSmootherConfig(record_smoothed_states_mean=True)):
    out = Predictive(model, params={"rho": jnp.array(0.3)}, num_samples=1,
                     exclude_deterministic=False)(
        jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values)

mll = out["f_marginal_loglik"]
smoothed_means = out["f_smoothed_states_mean"]   # (num_samples, T, state_dim)
```

A smoother also returns `f_marginal_loglik`, so it can drive parameter inference exactly as a filter does. Overlaying the smoothed mean on the true simulated states is the correctness signal for a smoothing task. For online estimates or for the pseudo-marginal likelihood alone, use a filter from [inference-filtering.md](inference-filtering.md) instead.
