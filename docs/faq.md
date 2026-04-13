# Frequently Asked Questions

## What is the `ObservationModel` class / do I need it?

`ObservationModel` is a convenience class that wraps a callable `(x, u, t) -> Distribution` into a standard interface with `log_prob` and `sample` methods. You don't strictly need it: you can pass any callable that returns a NumPyro distribution to `DynamicalModel`'s `observation_model` argument. The built-in `LinearGaussianObservation` and `DiracIdentityObservation` implement this interface for common cases. See the [observations API reference](api_reference/public/models/core/observation_model.md) for details.

## What are the most common ways to condition models on data for system identification?

Say you have a dynestyx model `model` that accepts `obs_times`, `obs_values` (and optionally `ctrl_times`, `ctrl_values` for controlled systems) and passes them to `dsx.sample`:
```python
def model(obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None):
    params = numpyro.sample(...)
    dynamics = DynamicalModel(...)
    return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values, ctrl_times=ctrl_times, ctrl_values=ctrl_values)
```
Omit `ctrl_times` and `ctrl_values` when the model has no controls.


- **HMM**: Use the HMM filter (using an `HMMConfig` configuation). See [HMM inference](tutorials/gentle_intro/07_hmm.ipynb).
```python
from dynestyx.inference.filters import Filter, HMMConfig

with Filter(filter_config=HMMConfig()):
    return model(obs_times=obs_times, obs_values=obs_values)
```

- **Discrete-time**: Either a **Simulator** (NUTS samples both parameters and latent states) or a **Filter** (pseudo-marginal MCMC—parameters only). Note: the usage of discrete-time filters is currently under active development (likely incorrect implementations).
For explicit representation of latent states (NUTS / SVI do all the work of parameter and latent state inference), use the simulator approach (currently working reliably), do:
```python
with DiscreteTimeSimulator():
    return model(obs_times=obs_times, obs_values=obs_values)
```
For filter-based marginalization (currently not working reliably), do:
```python
with Filter():
    return model(obs_times=obs_times, obs_values=obs_values)
```


- **Continuous-time stochastic differential equation**: **Filter** is the main choice. EnKF is the default and works well for nonlinear models, but only works if your initial condition and observation model are linear/gaussian. Use the particle filter (PF) only if you have non-Gaussian initial conditions or observation models—see [SDE with non-Gaussian observations](tutorials/sde_non_gaussian_observations.ipynb). We stand by these implementations, and they appear to be working well currently (especially EnKF).
```python
from dynestyx.inference.filters import Filter, ContinuousTimeEnKFConfig, ContinuousTimeDPFConfig

with Filter(filter_config=ContinuousTimeEnKFConfig()):
# with Filter(filter_config=ContinuousTimeDPFConfig(n_particles=1000)):
    return model(obs_times=obs_times, obs_values=obs_values)
```

If you happen to have high-frequency, fully-observed, low-noise data, then there IS a much faster option, as shown in this [deep dive](deep_dives/l63_speedup_dirac_vs_enkf.ipynb). Simply do:
```python
with DiscreteTimeSimulator():
    with Discretizer():
        return model(obs_times=obs_times, obs_values=obs_values, dirac_observation=True)
```

- **Continuous-time ordinary differential equation**: You can use a **Simulator** or a **Filter**. The simulator simply rolls out solutions from the initial conditions and checks fit to data; see tutorial on [ODE inference](tutorials/gentle_intro/06b_odes.ipynb).
```python
with ODESimulator():
    return model(obs_times=obs_times, obs_values=obs_values)
```
Despite the deterministic nature of an ODE, sometimes a filtering-algorithm helps a lot (especially for long timeseries rollouts, partial/noisy observations, systems with large sensitivities to intial conditions). You can modify the model definition to have a small diffusion coefficient to "relax" the ODE problem to an SDE.
```python
from dynestyx.inference.filters import Filter, ContinuousTimeEnKFConfig

with Filter(filter_config=ContinuousTimeEnKFConfig()):
    return model(obs_times=obs_times, obs_values=obs_values, diffusion_coefficient=0.01)
```




## What about multiple trajectories?

All three simulators support generating multiple independent trajectories in a single call via the `n_simulations` parameter:

```python
from dynestyx import SDESimulator, flatten_draws
from numpyro.infer import Predictive

with SDESimulator(n_simulations=100):
    samples = Predictive(model, num_samples=1)(jr.PRNGKey(0), predict_times=times)

# f_states shape: (num_samples=1, n_sim=100, T, state_dim)
# flatten_draws merges the (num_samples, n_sim) prefix into one axis:
states = flatten_draws(samples["f_states"])  # (100, T, state_dim)
```

The `n_simulations` parameter is available on `DiscreteTimeSimulator`, `SDESimulator`, and `ODESimulator`.

**Shape contract:** all trajectory outputs have shape `(num_samples, n_sim, T, dim)` — a leading `num_samples` axis from `Predictive`, a `n_sim` axis from the simulator, the time axis `T`, and then the state/observation dimension. Use `dynestyx.flatten_draws` to collapse the first two axes into a single draws axis for plotting or analysis.

**Note:** conditioning on observations (`obs_values != None`) is only supported with `n_simulations=1`. For filter-based rollouts from a posterior (Filter + predict_times), `n_simulations > 1` is fully supported.

## What about hierarchical models?

Hierarchical models are supported by the [`dsx.plate`](./api_reference/public/handlers.md) primitive! This allows for multiple levels of hierarchy (e.g., modelling populations, treatment arms, and individuals within each treatment arm), or simple multi-trajectory inference. You can see an example [here](./tutorials/gentle_intro/08_hierarchical_inference.ipynb).

## What about neural nets?

We will put examples up soon. See [CD-Dynamax's Lorenz 63 neural drift tutorial](https://github.com/hd-UQ/cd_dynamax/blob/dev-numpyro-api/demos/numpyro/notebooks/lorenz63_nndrift_sgd_fit_to_data_tutorial_newAPI.ipynb) to convince yourself that this will work well.

## What about SINDy?

See our [Sparse system identification deep dive](deep_dives/fhn_sparse_id.ipynb). TL;DR: pick a Laplace or Spike-and-Slab prior and do everything else the dynestyx-way.

## Why are particle filters underperforming?

Yes, they are worse than we thought in pseudo-marginal settings too. This is an area of active research. If you know how to do things better, please tell us!

## How can I contribute?

Open an [issue](https://github.com/BasisResearch/dynestyx/issues) or submit a [Pull Request](https://github.com/BasisResearch/dynestyx/pulls) on GitHub.
