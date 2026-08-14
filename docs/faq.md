# Frequently Asked Questions

## What is the `ObservationModel` class / do I need it?

`ObservationModel` is a convenience class that wraps a callable `(x, u, t) -> Distribution` into a standard interface with `log_prob` and `sample` methods. You don't strictly need it: you can pass any callable that returns a NumPyro distribution to `DynamicalModel`'s `observation_model` argument. The built-in `LinearGaussianObservation` and `DiracIdentityObservation` implement this interface for common cases. See the [observations API reference](api_reference/public/models/core/observation_model.md) for details.

## Why isn't the math rendering on this website?

We don't know, but usually a refresh fixes it :)

## What are the most common ways to condition models on data for system identification?

Say you have a dynestyx model `model` that accepts `obs_times`, `obs_values`
(and optionally controls and prediction times) and passes them to `dsx.sample`:

```python
def model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
):
    params = numpyro.sample(...)
    dynamics = dsx.DynamicalModel(...)
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
    )
```

Omit `ctrl_times` and `ctrl_values` when the model has no controls.

- **HMM**: Use the HMM filter with an `HMMConfig`.

```python
from dynestyx.inference.filters import HMMConfig

with dsx.Filter(filter_config=HMMConfig()):
    model(obs_times=obs_times, obs_values=obs_values)
```

  See the [HMM inference tutorial](tutorials/gentle_intro/07_hmm.ipynb).

- **Discrete-time**: Use `LatentPathBuilder` for explicit joint inference over
  parameters and latent states, or a `Filter` to marginalize the latent states
  while inferring parameters. `dsx.Filter()` defaults to an EnKF for a
  discrete-time model; pass a filter config to select another algorithm.

```python
# Explicit latent path
with dsx.LatentPathBuilder():
    model(obs_times=obs_times, obs_values=obs_values)

# Marginalized latent path
with dsx.Filter():
    model(obs_times=obs_times, obs_values=obs_values)
```

  See the [filtering and marginal-likelihood
  tutorial](tutorials/gentle_intro/03_filtering_mll.ipynb).

- **Continuous-time stochastic differential equation**: Use a `Discretizer`
  with `LatentPathBuilder` for explicit latent-state inference, or a `Filter`
  for marginalized inference. The continuous-time EnKF is the default filter.

```python
# Explicit latent path
with dsx.LatentPathBuilder():
    with dsx.Discretizer(discretize=dsx.euler_maruyama):
        model(obs_times=obs_times, obs_values=obs_values)

# Marginalized latent path
with dsx.Filter():
    model(obs_times=obs_times, obs_values=obs_values)
```

  Use a particle filter for non-Gaussian observations;
  see [SDE inference with non-Gaussian
  observations](tutorials/sde_non_gaussian_observations.ipynb). See the
  [continuous-time
  tutorial](tutorials/gentle_intro/06_continuous_time.ipynb) for the full
  workflow.

  With high-frequency, low-noise data, the explicit-path approach can be
  especially attractive when it is reasonable to treat measurements as exact
  using `dsx.DiracIdentityObservation`, dramatically simplifying inference and
  improving its efficiency. For example, if the model selects that observation
  model when `dirac_observation=True`:

```python
with dsx.LatentPathBuilder():
    with dsx.Discretizer(discretize=dsx.euler_maruyama):
        model(
            obs_times=obs_times,
            obs_values=obs_values,
            dirac_observation=True,
        )
```

  In this Dirac setting, observed state coordinates are fixed exactly while
  unobserved or missing coordinates remain latent. Full observation is not
  required, although the largest speedups occur when most of the path is
  observed. See the [Discretizer
  reference](api_reference/public/discretizers/discretizer.md) and the
  [Dirac-observation deep dive](deep_dives/l63_speedup_dirac_vs_enkf.ipynb).

- **Continuous-time ordinary differential equation**: Use
  `LatentPathBuilder` for explicit latent-state inference or a `Filter` for
  marginalized inference.

```python
# Explicit latent path
with dsx.LatentPathBuilder():
    model(obs_times=obs_times, obs_values=obs_values)

# Marginalized latent path
with dsx.Filter():
    model(obs_times=obs_times, obs_values=obs_values)
```

Despite the deterministic nature of an ODE, sometimes a filtering-algorithm helps a lot (especially for long timeseries rollouts, partial/noisy observations, systems with large sensitivities to intial conditions). Continuous-time filters work directly with diffusion equal to zero, but you can modify the model definition to have a small diffusion coefficient to "relax" the ODE problem to an SDE. See the [ODE inference tutorial](tutorials/gentle_intro/06b_odes.ipynb).



You can modify the model definition to have a small diffusion coefficient to "relax" the ODE problem to an SDE.

Finally, wrap any `dsx.Filter()` configuration around MCMC to infer the
parameters in `model`:

```python
import jax.random as jr
from dynestyx.inference.configs.mcmc import NUTSConfig
from dynestyx.inference.mcmc import MCMCInference

with dsx.Filter(filter_config=my_filter_config):
    inference = MCMCInference(
        mcmc_config=NUTSConfig(
            num_samples=1_000,
            num_warmup=1_000,
            num_chains=1,
            mcmc_source="numpyro",
        ),
        model=model,
    )
    posterior_samples = inference.run(
        rng_key=jr.PRNGKey(0),
        obs_times=obs_times,
        obs_values=obs_values,
    )
```

The filter supplies the marginal likelihood used by MCMC, so the sampler
targets model parameters without explicitly sampling the latent state path.

You can also use NumPyro's MCMC classes directly:

```python
from numpyro.infer import MCMC, NUTS

with dsx.Filter(filter_config=my_filter_config):
    mcmc = MCMC(
        NUTS(model),
        num_warmup=1_000,
        num_samples=1_000,
    )
    mcmc.run(
        jr.PRNGKey(0),
        obs_times=obs_times,
        obs_values=obs_values,
    )
    posterior_samples = mcmc.get_samples()
```

See the [filtering with NUTS
tutorial](tutorials/gentle_intro/04_filtering_nuts_pseudomarginal.ipynb) and
the [`MCMCInference` API
reference](api_reference/public/inference/mcmc.md).

## What if my data has missingness?

All primary inference workflows support missing data. Use `jnp.nan` for missing
entries in `obs_values`; filters, smoothers, `LatentPathBuilder`, HMM inference,
and direct `dsx.log_prob` scoring can handle fully missing time points and
partially missing observation coordinates. The exact treatment and compatible
backend depend on the observation distribution.

See the tutorials on missing observations with [filters and
smoothers](tutorials/gentle_intro/11_missing_observations.ipynb),
[`LatentPathBuilder`](tutorials/gentle_intro/11b_missing_observations_latent_path_mcmc.ipynb),
and [HMMs](tutorials/gentle_intro/11c_missing_observations_hmms.ipynb).

## How do I simulate multiple trajectories?

For a concrete `dynamics` object, use
[`dsx.simulate(...)`](api_reference/public/handlers.md#dynestyx.api.simulate)
and set
`n_simulations`:

```python
import jax.random as jr

result = dsx.simulate(
    dynamics,
    rng_key=jr.PRNGKey(0),
    predict_times=times,
    n_simulations=100,
)
states = result.states  # (100, T, state_dim)
```

This returns a
[`SimulatedResult`](api_reference/public/result_types.md#dynestyx.types.SimulatedResult).
`dsx.simulate` auto-routes to the discrete-time, ODE, or SDE backend. For a
differential equation, pass an
[`ODESimulatorConfig` or
`SDESimulatorConfig`](api_reference/public/simulators/simulator_configs.md)
through `simulator_config`. The same `n_simulations` contract holds for all
three lower-level simulators.

For prior or posterior `Predictive` workflows, use the
[`dsx.Simulator`](api_reference/public/simulators/simulator_wrapper.md) handler
in the same way:

```python
from numpyro.infer import Predictive

with dsx.Simulator(n_simulations=100):
    samples = Predictive(
        model, num_samples=1, exclude_deterministic=False
    )(jr.PRNGKey(0), predict_times=times)
```

Here `n_simulations` draws trajectories conditional on each parameter
realization, while `Predictive(num_samples=...)` controls the number of NumPyro
model executions. The corresponding trajectory shape is
`(num_samples, n_simulations, T, *event_shape)`.

## What if I do not want to use NumPyro or `Predictive`?

Call the result-returning APIs directly on a `dynamics` object constructed at
concrete parameter values. For simulation, use
[`dsx.simulate(...)`](api_reference/public/handlers.md#dynestyx.api.simulate)
as shown above; filtering, smoothing, and latent-path scoring use:

```python
# Auto-routed filtering and its marginal log likelihood.
with dsx.Filter():
    filtered = dsx.condition(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )

# Auto-routed smoothing and its marginal log likelihood.
with dsx.Smoother():
    smoothed = dsx.condition(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )

# Joint density of fixed latent-state parameters and observations.
joint_log_prob = dsx.log_prob(
    dynamics,
    state_path_params=state_path_params,
    state_path_param_times=state_path_param_times,
    obs_times=obs_times,
    obs_values=obs_values,
)
```

`dsx.condition` returns a
[`ConditionedResult`](api_reference/public/result_types.md#dynestyx.types.ConditionedResult)
containing the inference times, marginal log likelihood, state summaries, and
per-time distributions requested by the active filter or smoother handler.
Pass a filter or smoother config when you need a specific algorithm.
`dsx.log_prob` returns the joint density of a fixed latent path. For a native
SDE latent-path workflow, first choose a
[`Discretizer`](api_reference/public/discretizers/discretizer.md); see the
[discretized latent-path
tutorial](deep_dives/l63_speedup_dirac_vs_enkf.ipynb).

These APIs are compatible with `jax.jit`, `jax.vmap`, and `jax.grad` when the
selected algorithm is itself differentiable. They do not provide NumPyro
priors or parameter inference: construct `dynamics` at concrete parameter
values and use the optimizer or inference library of your choice.

For NumPyro-free system identification, put an optimizer or sampler around a
loss that constructs `dynamics` from the current parameters. With marginalized
latent states, minimize the negative marginal log likelihood returned by a
filter:

```python
def filter_loss(parameters):
    dynamics = build_dynamics(parameters)
    with dsx.Filter(filter_config=my_filter_config):
        result = dsx.condition(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
        )
    return -result.marginal_loglik


# Pseudocode: use an optimizer such as Optax or SciPy.
parameters = optimizer_loop(filter_loss, initial_parameters)
```

For explicit latent-path system identification, optimize or sample parameters
and the latent path using `dsx.log_prob`:

```python
def latent_path_loss(parameters, state_path):
    dynamics = build_dynamics(parameters)
    return -dsx.log_prob(
        dynamics,
        state_path_params=state_path,
        state_path_param_times=state_times,
        obs_times=obs_times,
        obs_values=obs_values,
    )


# Pseudocode: use the optimizer or sampler of your choice.
parameters, state_path = optimizer_or_sampler_loop(
    latent_path_loss,
    initial_parameters,
    initial_state_path,
)
```

See the [NumPyro-free filtering and marginal-likelihood
tutorial](tutorials/gentle_intro/03_filtering_mll_no_numpyro.ipynb), the
[direct filtering and smoothing
example](tutorials/state_space_models/kf_tracking.ipynb), and the
[NumPyro-free differentiable optimization
tutorial](tutorials/gentle_intro/05_svi_no_numpyro.ipynb). The [result-type
reference](api_reference/public/result_types.md) documents `SimulatedResult`
and `ConditionedResult`.

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
