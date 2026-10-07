# Handlers

`dynestyx` is built using [`effectful`](https://github.com/BasisResearch/effectful),
which interprets primitives according to the active handler stack.

The main user-facing primitives are:

- `sample(...)`: effectful primitive used inside handler stacks
- `condition(...)`: pure result-returning conditioning entry point
- `simulate(...)`: pure-JAX forward simulation entry point
- `log_prob(...)`: pure-JAX joint trajectory scoring entry point
- `plate(...)`: hierarchical batching primitive

For **hierarchical** models with multiple trajectories, use `plate` together
with NumPyro sampling inside the plate context. For example,

```python
with Filter(EKFConfig()):
    dsx.sample("f", dynamical_model, obs_times=obs_times, obs_values=obs_values)
```

will implement the `dsx.sample` primitive using an extended Kalman filter. For more details, see the corresponding [developer API page](../developer/handlers.md).

## Handler primitives

::: dynestyx.handlers
    options:
        members:
            - condition
            - sample
            - plate

## Pure APIs

::: dynestyx.api
    options:
        members:
            - log_prob
            - simulate

## Gaussian relaxation

`GaussianRelaxation` converts selected discrete-time components into Gaussian
models. Start with a NumPyro `Delta` initial condition,
`DeterministicStateEvolution(F)`, and/or `DeterministicObservation(h)` (or
`DiracIdentityObservation()`). Existing Gaussian components also support
covariance replacement or addition. Deterministic components return Delta
distributions with scalar events or trailing vector event axes.

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

Each setting defaults to `None`, leaving that component unchanged. Scalar
shorthand is a **variance**, vector shorthand contains **diagonal
variances**, and matrix shorthand is a **full covariance**. Explicit
`ScalarCovariance(sd=... | variance=...)`,
`DiagonalCovariance(sd=... | variance=...)`, and `FullCovariance(matrix)` objects
also work in existing Gaussian constructors and `LTI_discrete`. Use these
objects for batched settings: leading axes are plate axes, while trailing
covariance event axes have rank zero, one, or two respectively.

`mode="replace"` replaces the selected covariances. `mode="add"` adds the
specified covariance to each selected component's covariance, treating
deterministic components as having zero covariance. One mode applies to all
selected components.

For example, if a `GaussianStateEvolution` already defines a covariance function
`cov(x, u, t_now, t_next)`, adding covariance `C` gives
`cov(x, u, t_now, t_next) + C` at each transition. This preserves the original
function's dependence on state, controls, and time. Time-varying linear-model
functions `cov(t_now, t_next)` and `R(t)` work the same way. The noise settings
passed to `GaussianRelaxation` accept constants or learned JAX/NumPyro arrays,
expressed as scalars, vectors, matrices, or covariance objects.

Relaxation uses the specified model covariance exactly. Covariance objects
check shape, finite values, and symmetry. Supply positive-semidefinite noise
covariances and a positive-definite effective covariance for Gaussian densities.
Zero covariance can be added to an already positive-definite Gaussian. The EnKF setting
`recorded_filtered_states_cov_jitter` continues to add jitter to the returned
state distributions; it leaves the filter recursion and marginal likelihood
unchanged.

Use the nesting order, outermost first:
`Evaluation`, `Simulator`, `Filter`/`Smoother`/`LatentPathBuilder`,
`GaussianRelaxation`, `Discretizer`, `plate`. Omit unused stages. Only one
relaxation handler may be active. The relaxer accepts only discrete-time models
and raises `TypeError` otherwise.
Selected transitions/observations must be the deterministic or Gaussian model
classes. Existing linear-Gaussian classes retain their structure and filter
eligibility.

For direct simulation or scoring, use the same transformation explicitly:

```python
relaxed = dsx.relax_dynamics(dynamics, observation_model_cov=0.1)
draws = dsx.simulate(relaxed, rng_key=jax.random.key(0), predict_times=jnp.arange(3.0))
```

Both entry points preserve everything other than the components you ask to
change.

For exact identity measurements (`y = x`), use `DiracIdentityObservation()`.
With `LatentPathBuilder`, observed values fix those state coordinates, and only
missing state coordinates are inferred. Relaxing the observation component
turns these into noisy Gaussian measurements: observed values contribute a
likelihood, and the latent state coordinates are inferred.

::: dynestyx.relaxations
    options:
        members:
            - GaussianRelaxation
            - relax_dynamics
