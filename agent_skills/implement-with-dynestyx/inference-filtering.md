# Filtering

Filters compute the filtering distribution over the latent state given past observations and return the marginal log-likelihood `f_marginal_loglik`. Read [reference.md](reference.md) first.

## Choosing a filter

Match the filter to the model.

- Linear dynamics with Gaussian noise → `KFConfig`, the exact Kalman filter.
- Nonlinear dynamics, roughly Gaussian → `EKFConfig` for a Jacobian linearization or `UKFConfig` for a derivative-free sigma-point version.
- Nonlinear dynamics, higher dimension, ensemble approach → `EnKFConfig`. A bare `Filter()` defaults to this.
- Nonlinear and non-Gaussian, arbitrary likelihoods, tracking → `PFConfig`, the bootstrap particle filter. The right default whenever the observation model is not Gaussian, which is the common reason a Kalman variant fails.
- Discrete latent regimes → `HMMConfig`.

Every Gaussian filter and the particle filter has a continuous-time variant for models with an SDE between observations, named `ContinuousTimeKFConfig`, `ContinuousTimeEKFConfig`, `ContinuousTimeUKFConfig`, `ContinuousTimeEnKFConfig`, and `ContinuousTimeDPFConfig`. Use these when the dynamics are continuous in time and you do not want to discretize first.

Import configs from `dynestyx.inference.filters`. `Filter` is exported from the top level.

## Likelihood sweeps and recording states

Wrap the conditioning pattern from [reference.md](reference.md) in a function and `vmap` it to profile the marginal log-likelihood over a parameter. The argmax is a quick parameter estimate and a good first correctness check on simulated data.

```python
import jax.random as jr
from jax import vmap
from numpyro.infer import Predictive
from dynestyx import Filter
from dynestyx.inference.filters import EKFConfig, PFConfig

def get_mll(rho, filter_config):
    with Filter(filter_config):
        out = Predictive(model, params={"rho": rho}, num_samples=1,
                         exclude_deterministic=False)(
            jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values)
    return out["f_marginal_loglik"]

rho_grid = jnp.linspace(-0.8, 0.8, 50)
mll_pf = vmap(lambda r: get_mll(r, PFConfig(n_particles=1000)))(rho_grid)
rho_hat = rho_grid[jnp.argmax(mll_pf)]
```

Filtered state estimates are recorded on request and then read from the trace.

```python
with Filter(filter_config=EKFConfig(record_filtered_states_mean=True)):
    out = Predictive(model, params={"rho": jnp.array(0.3)}, num_samples=1,
                     exclude_deterministic=False)(
        jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values)
filtered_means = out["f_filtered_states_mean"]   # (num_samples, T, state_dim)
```

## Particle filter notes

`PFConfig(n_particles=...)` runs the bootstrap particle filter and handles arbitrary nonlinear dynamics and non-Gaussian observation models. Set `n_particles` from the difficulty of the problem, in the thousands for hard tracking. Resampling is configured through `PFResamplingConfig`. For a continuous-time SDE with discrete observations, use `ContinuousTimeDPFConfig` instead.

## Forecasting

To roll a fitted model forward, nest a simulator outside the filter and pass `predict_times`. The filter conditions on the training window and the simulator predicts beyond it.

```python
from dynestyx import DiscreteTimeSimulator

with DiscreteTimeSimulator(n_simulations=20):
    with Filter(filter_config=EKFConfig(record_filtered_states_mean=True)):
        out = Predictive(model, params={"rho": rho_hat}, num_samples=1,
                         exclude_deterministic=False)(
            jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values,
            ctrl_times=ctrl_times_full, ctrl_values=ctrl_values_full,
            predict_times=future_times)
pred_states = out["f_predicted_states"]   # (num_samples, n_sim, T_pred, state_dim)
```
