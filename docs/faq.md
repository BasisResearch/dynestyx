# Frequently Asked Questions

## What is the `ObservationModel` class / do I need it?

`ObservationModel` is a convenience class that wraps a callable `(x, u, t) -> Distribution` into a standard interface with `log_prob` and `sample` methods. You don't strictly need it: you can pass any callable that returns a NumPyro distribution to `DynamicalModel`'s `observation_model` argument. The built-in `LinearGaussianObservation` and `DiracIdentityObservation` implement this interface for common cases. See the [observations API reference](api_reference/observations.md) for details.

## What are the most common ways to condition models on data for system identification?

Say you have a dynestyx model `model`:
```python

def model(...):
    params = numpyro.sample(...)
    dynamics = DynamicalModel(...)
    return dsx.sample("f", dynamics)
```
and data `context = Context(observations=Trajectory(times=times, values=values), controls=...)`.


- **HMM**: Use the HMM filter (`FilterBasedHMMMarginalLogLikelihood`). See [HMM inference](tutorials/hmm_inference.ipynb).
```python
with FilterBasedHMMMarginalLogLikelihood():
    with Condition(context):
        return model()
```

- **Discrete-time**: Either a **Simulator** (NUTS samples both parameters and latent states) or a **Filter** (pseudo-marginal MCMC—parameters only). Note: the usage of discrete-time filters is currently under active development (likely incorrect implementations).
For explicit representation of latent states (NUTS / SVI do all the work of parameter and latent state inference), use the simulator approach (currently working reliably), do:
```python
with DiscreteTimeSimulator():
    with Condition(context):
        return model()
```
For filter-based marginalization (currently not working reliably), do:
```python
with FilterBasedMarginalLogLikelihood():
    with Condition(context):
        return model()
```


- **Continuous-time stochastic differential equation**: **Filter** is the main choice. EnKF is the default and works well for nonlinear models, but only works if your initial condition and observation model are linear/gaussian. Use the particle filter (PF) only if you have non-Gaussian initial conditions or observation models—see [SDE with non-Gaussian observations](tutorials/sde_non_gaussian_observations.ipynb). We stand by these implementations, and they appear to be working well currently (especially EnKF).
```python
with FilterBasedMarginalLogLikelihood(filter_type='enkf'):
# with FilterBasedMarginalLogLikelihood(filter_type='pf'): 
    with Condition(context):
        return model()
```
 EnKF is the default and works well for nonlinear models, but only works if your initial condition and observation model are linear/gaussian. Use the particle filter (PF) only if you have non-Gaussian initial conditions or observation models—see [SDE with non-Gaussian observations](tutorials/sde_non_gaussian_observations.ipynb). We stand by these implementations, and they appear to be working well currently (especially EnKF).

If you happen to have high-frequency, fully-observed, low-noise data, then there IS a much faster option, as shown in this [deep dive](deep_dives/l63_speedup_dirac_vs_enkf.ipynb). Simply do:
```python
with DiscreteTimeSimulator():
    with Discretizer():
        with Condition(context):
            return model(dirac_observation=True)
```

- **Continuous-time ordinary differential equation**: You can use a **Simulator** or a **Filter**. The simulator simply rolls out solutions from the initial conditions and checks fit to data; see tutorial on [ODE inference](tutorials/ode_inference.ipynb).
```python
with ODESimulator():
    with Condition(context):
        return model()
```
Despite the deterministic nature of an ODE, sometimes a filtering-algorithm helps a lot (especially for long timeseries rollouts, partial/noisy observations, systems with large sensitivities to intial conditions). You can modify the model definition to have a small diffusion coefficient to "relax" the ODE problem to an SDE.
```python
with FilterBasedMarginalLogLikelihood(filter_type='enkf'):
    with Condition(context):
        return model(diffusion_coefficient = 0.01)
```




## What about multiple trajectories?

Feature coming soon.

## What about hierarchical models?

Feature coming soon.

## What about neural nets?

We will put examples up soon. See [CD-Dynamax's Lorenz 63 neural drift tutorial](https://github.com/hd-UQ/cd_dynamax/blob/dev-numpyro-api/demos/numpyro/notebooks/lorenz63_nndrift_sgd_fit_to_data_tutorial_newAPI.ipynb) to convince you that this will work well.

## What about SINDy?

See our [Sparse system identification deep dive](deep_dives/fhn_sparse_id.ipynb). TL;DR: pick a Laplace or Spike-and-Slab prior and do everything else the dynestyx-way.

## Why are particle filters underperforming?

Yes, they are worse than we thought in pseudo-marginal settings too. This is an area of active research. If you know how to do things better, please tell us!

## How can I contribute?

Open an [issue](https://github.com/BasisResearch/dynestyx/issues) or submit a [Pull Request](https://github.com/BasisResearch/dynestyx/pulls) on GitHub.
