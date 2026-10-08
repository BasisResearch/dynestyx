# Gaussian relaxation

`GaussianRelaxation` adds Gaussian noise to a deterministic or Gaussian component of a `DynamicalModel`. In particular, `Delta`/`Normal`/`MultivariateNormal` initial conditions can be relaxed, as can a `DeterministicStateEvolution`/`GaussianStateEvolution` and `DeterministicObservation`/`DiracIdentityObservation`/`GaussianObservation`.

Relaxation requires a discrete-time model. Selected transitions and observations
must use the deterministic or Gaussian model classes; linear-Gaussian classes
retain their structure and filter eligibility.

There are two ways to apply the same transformation:

- **Effect-handler form**: use `GaussianRelaxation` around a model call so
  inference or simulation receives the relaxed model.
- **Direct form**: call `relax_dynamics` to obtain a model for direct simulation,
  scoring, or use in your own inference code.

## Effect-handler form

Use `GaussianRelaxation` with the [handler order](../handlers.md#handler-order):

```python
import dynestyx as dsx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from dynestyx.inference.configs.filter import EnKFConfig

dynamics = dsx.DynamicalModel(
    initial_condition=dist.Delta(jnp.array([1.0, 2.0]), event_dim=1),
    state_evolution=dsx.DeterministicStateEvolution(
        lambda x, u, t_now, t_next: x + (t_next - t_now) * jnp.sin(x)
    ),
    observation_model=dsx.DeterministicObservation(lambda x, u, t: x**2),
)

with dsx.Filter(EnKFConfig()), dsx.GaussianRelaxation(
    initial_condition_cov=0.05,
    state_evolution_cov=dsx.DiagonalCovariance(variance=jnp.array([0.01, 0.02])),
    observation_model_cov=dsx.FullCovariance(0.1 * jnp.eye(2)),
):
    result = dsx.condition(
        "trajectory", dynamics,
        obs_times=jnp.arange(3.0),
        obs_values=jnp.array([[1.0, 4.0], [3.4, 8.5], [7.9, 9.9]]),
    )
```

## Direct form

Using the model above, call `relax_dynamics` before direct simulation or scoring:

```python
relaxed = dsx.relax_dynamics(dynamics, observation_model_cov=0.1)
draws = dsx.simulate(relaxed, rng_key=jax.random.key(0), predict_times=jnp.arange(3.0))
```

Both entry points preserve everything other than the components you ask to
change.

## Selecting components and noise

Each setting defaults to `None`, leaving that component unchanged. See
[covariance specifications](../models/specialized_models.md#covariance-specifications)
for accepted representations, units, and batched settings.

`mode="replace"` replaces the selected covariances. `mode="add"` adds the
specified covariance to each selected component's covariance, treating
deterministic components as having zero covariance. One mode applies to all
selected components.

For example, if a `GaussianStateEvolution` already defines a covariance function
`cov(x, u, t_now, t_next)`, adding covariance `C` gives
`cov(x, u, t_now, t_next) + C` at each transition. Time-varying linear-model
functions `cov(t_now, t_next)` and `R(t)` work the same way. The added noise
settings accept constants or learned JAX/NumPyro arrays, not callables.

## API reference

::: dynestyx.relaxations
    options:
      members:
        - GaussianRelaxation
        - relax_dynamics
      show_root_heading: false
      show_root_toc_entry: false

See the [developer API](../../developer/relaxations/gaussian_relaxation.md)
for implementation details.
