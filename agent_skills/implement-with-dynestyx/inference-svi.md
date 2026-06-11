# SVI

SVI fits an approximate posterior by optimizing a variational guide. It trades exactness for speed and scale. Read [reference.md](reference.md) first. The handler around the model decides what is inferred.

## Filter-conditioned SVI

Define a zero-argument wrapper that conditions the model under a `Filter`, build an autoguide over it, and optimize the ELBO. The guide then approximates the posterior over the parameters with the states marginalized.

```python
import jax.random as jr
import optax
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoMultivariateNormal
from dynestyx import Filter

def filter_conditioned_model():
    with Filter():   # ensemble Kalman filter by default
        return model(obs_times=obs_times, obs_values=obs_values,
                     ctrl_times=ctrl_times, ctrl_values=ctrl_values)

guide = AutoMultivariateNormal(filter_conditioned_model)
optimizer = optax.adam(learning_rate=1e-3)
svi = SVI(filter_conditioned_model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(jr.PRNGKey(1), num_steps=1500)

posterior = Predictive(guide, params=svi_result.params, num_samples=500)(jr.PRNGKey(2))
```

Pick the filter inside the wrapper by the rules in [inference-filtering.md](inference-filtering.md).

## Joint SVI

Swap the filter for a simulator in the wrapper to put the latent states in the guide and infer states and parameters jointly.

```python
from dynestyx import DiscreteTimeSimulator

def joint_conditioned_model():
    with DiscreteTimeSimulator():
        return model(obs_times=obs_times, obs_values=obs_values,
                     ctrl_times=ctrl_times, ctrl_values=ctrl_values)

guide = AutoMultivariateNormal(joint_conditioned_model)
svi = SVI(joint_conditioned_model, guide, optimizer, loss=Trace_ELBO())
svi_result = svi.run(jr.PRNGKey(3), num_steps=15000)
```

## After fitting

Sample the fitted guide with `Predictive(guide, params=svi_result.params, num_samples=...)` and read parameter draws by name. Recovering the known generating value on simulated data is the correctness signal. Watch the ELBO trace in `svi_result.losses` for convergence, and raise `num_steps` or lower the learning rate when it has not flattened.
