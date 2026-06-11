# MCMC

MCMC draws a posterior over the unknown parameters. Read [reference.md](reference.md) first. There are two modes, and the only difference is the handler you wrap around the model.

## Pseudo-marginal MCMC

Wrap the model in a `Filter`. The filter integrates out the latent states and contributes the marginal log-likelihood, so NUTS explores the parameters with the states marginalized. This is the method for nonlinear or non-Gaussian models where the states are nuisance variables.

```python
import jax.random as jr
from numpyro.infer import MCMC, NUTS
from dynestyx import Filter
from dynestyx.inference.filters import EKFConfig

with Filter(filter_config=EKFConfig()):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=100)
    mcmc.run(jr.PRNGKey(1), obs_times=obs_times, obs_values=obs_values,
             ctrl_times=ctrl_times, ctrl_values=ctrl_values)
posterior = mcmc.get_samples()
```

Choose the filter by the rules in [inference-filtering.md](inference-filtering.md). Use a particle filter here for non-Gaussian likelihoods, which gives a particle-marginal Metropolis-Hastings style sampler.

## Joint state and parameter MCMC

Wrap the model in a simulator instead of a filter. The latent states stay in the trace, so NUTS samples states and parameters together. This suits linear-Gaussian or mildly nonlinear models where the joint posterior is well behaved.

```python
from dynestyx import DiscreteTimeSimulator

with DiscreteTimeSimulator():
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_warmup=100, num_samples=100)
    mcmc.run(jr.PRNGKey(1), obs_times=obs_times, obs_values=obs_values,
             ctrl_times=ctrl_times, ctrl_values=ctrl_values)
posterior = mcmc.get_samples()
```

## After sampling

Read parameter draws by name, as in `posterior["rho"]`. A posterior whose mass covers the known generating value is a clean correctness signal on simulated data. To forecast from the fitted model, take a posterior summary such as the mean and follow the forecasting pattern in [inference-filtering.md](inference-filtering.md).

Gradient-based and stochastic-gradient kernels are also available through `dynestyx.inference.mcmc_configs`, including `NUTSConfig`, `HMCConfig`, `MALAConfig`, and `SGLDConfig`. Start with NUTS and reach for these only when scale or geometry demands it.
