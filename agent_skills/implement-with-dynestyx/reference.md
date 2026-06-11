# dynestyx core reference

The spine for writing dynestyx code. Read this before any inference file.

## Contents

- The model function contract
- State evolution
- Observation models
- Initial condition and diffusions
- Simulate data
- Condition on data and run inference
- Output sites and shapes
- Hierarchical models with plates
- Gotchas

## The model function contract

A dynestyx model is an ordinary NumPyro model function. It samples unknown parameters with `numpyro.sample`, builds a `DynamicalModel`, and returns `dsx.sample(name, dynamics, ...)`. The same function is reused for simulation and for every inference method. Inference is chosen by the handler you wrap around it, not by editing the model.

```python
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import dynestyx as dsx
from dynestyx import DynamicalModel

def model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
):
    rho = numpyro.sample("rho", dist.Uniform(-0.5, 0.5))
    A = jnp.array([[0.0, 0.3], [rho, -0.2]])

    # Anything that depends on a sampled parameter MUST be built inside the
    # model, so the dependence is traced.
    state_evolution = lambda x, u, t_now, t_next: dist.MultivariateNormal(
        A @ x, 0.1**2 * jnp.eye(2)
    )
    observation_model = lambda x, u, t: dist.Normal(x[0], 0.1)

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=state_evolution,
        observation_model=observation_model,
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
    )
```

The name passed to `dsx.sample` becomes the prefix on every output site. With `"f"` you read `f_observations`, `f_states`, `f_times`, `f_marginal_loglik`, `f_filtered_states_mean`, `f_predicted_states`, and so on.

`dsx.sample` takes `obs_times` and `obs_values` for observations to condition on, `ctrl_times` and `ctrl_values` for known controls, and `predict_times` for times to roll forward and predict. Pass `predict_times` to generate data, pass `obs_times` and `obs_values` to condition.

## State evolution

Three ways to specify dynamics, in rising order of structure.

Plain callable. Discrete time maps `(x, u, t_now, t_next)` to a distribution. Continuous time supplies a drift `(x, u, t)`.

```python
# discrete time transition
state_evolution = lambda x, u, t_now, t_next: dist.MultivariateNormal(A @ x, Q)

# continuous time drift, wrapped with a diffusion for an SDE
from dynestyx import ContinuousTimeStateEvolution, ScalarDiffusion
state_evolution = ContinuousTimeStateEvolution(
    drift=lambda x, u, t: A @ x,
    diffusion=ScalarDiffusion(1.0, bm_dim=2),
)
```

Typed classes. `LinearGaussianStateEvolution(A, cov, B=None, bias=None)` for `x_next ~ N(A x + B u + bias, cov)`. `GaussianStateEvolution(F, cov)` for a nonlinear mean `F(x, u, t_now, t_next)`.

LTI factories. `LTI_discrete(A, Q, H, R, ...)` and `LTI_continuous(A, L, H, R, ...)` build a whole linear-Gaussian `DynamicalModel` in one call, including the observation model and a default standard-normal initial condition.

A continuous-time model with a diffusion is an SDE and runs under `SDESimulator`. A continuous-time model with no diffusion is an ODE and runs under `ODESimulator`. Both can also be discretized for filtering with `Discretizer`.

## Observation models

Plain callable mapping `(x, u, t)` to a distribution, as in `lambda x, u, t: dist.Normal(x[0], sigma)`. Non-Gaussian likelihoods such as Poisson counts or heavy-tailed noise go here. Typed options are `LinearGaussianObservation(H, R, D=None, bias=None)` for `y ~ N(H x + D u + bias, R)`, `GaussianObservation(h, R)` for a nonlinear mean, and `DiracIdentityObservation()` for noiseless direct observation of the state.

## Initial condition and diffusions

The initial condition is any NumPyro distribution over the state, commonly `dist.MultivariateNormal(loc, cov)`.

Diffusions set the SDE noise. `ScalarDiffusion(sigma, bm_dim=d)` gives isotropic noise, `DiagonalDiffusion(vec, bm_dim=d)` gives per-coordinate noise, and `FullDiffusion(matrix, bm_dim=k)` gives a general state-by-Brownian matrix. The coefficient is a constant or a callable `(x, u, t)`.

## Simulate data

Generate synthetic data by running the model under a simulator with parameters held fixed. Pass `predict_times` for the times to produce. Use `exclude_deterministic=False` so the output sites are returned.

```python
import jax.random as jr
from numpyro.infer import Predictive
from dynestyx import DiscreteTimeSimulator

predictive = Predictive(model, params={"rho": jnp.array(0.3)}, num_samples=1,
                        exclude_deterministic=False)
with DiscreteTimeSimulator():
    pred = predictive(jr.PRNGKey(0), predict_times=obs_times,
                      ctrl_times=ctrl_times, ctrl_values=ctrl_values)

obs_values = pred["f_observations"][0, 0]   # (T, obs_dim)
states_true = pred["f_states"][0, 0]        # (T, state_dim)
```

Pick the simulator to match the dynamics. `DiscreteTimeSimulator` for discrete time, `SDESimulator` for an SDE, `ODESimulator` for an ODE. Each accepts `n_simulations` to draw several trajectories at once.

## Condition on data and run inference

Conditioning happens by wrapping the model in `Filter` (or `Smoother`) and supplying `obs_times` and `obs_values`. The filter integrates out the latent states and adds the marginal log-likelihood to the trace as a `numpyro.factor`, which is what makes parameter inference work.

```python
from dynestyx import Filter
from dynestyx.inference.filters import EKFConfig

with Filter(filter_config=EKFConfig()):
    out = Predictive(model, params={"rho": jnp.array(0.3)}, num_samples=1,
                     exclude_deterministic=False)(
        jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values)
mll = out["f_marginal_loglik"]
```

To profile the likelihood over a parameter, `vmap` the call. The mcmc, svi, smoothing, and filtering files build on this same wrap-then-run shape.

## Output sites and shapes

With the `dsx.sample` name `"f"`, common sites are below. Simulator outputs carry leading `(num_samples, n_simulations, ...)` axes. Filter and smoother summaries carry a leading `(num_samples, ...)` axis.

- `f_observations` shape `(num_samples, n_sim, T, obs_dim)`
- `f_states` shape `(num_samples, n_sim, T, state_dim)`
- `f_times` shape `(num_samples, n_sim, T)`
- `f_marginal_loglik` the conditioned log-likelihood
- `f_filtered_states_mean` shape `(num_samples, T, state_dim)` when recorded
- `f_predicted_states` and `f_predicted_times` for forecasts at `predict_times`

`flatten_draws` merges the `(num_samples, n_sim, T, D)` axes into `(num_samples * n_sim, T, D)` for plotting.

## Hierarchical models with plates

`dsx.plate(name, size)` batches several trajectories or groups. Sample shared or group-level parameters inside the plate, then call `dsx.sample` inside it. Observation arrays gain a leading group axis of `(N, T, obs_dim)`.

## Gotchas

- Build anything that depends on a sampled parameter inside the model function, never at module scope, or the dependence is not traced.
- Pass `exclude_deterministic=False` to `Predictive` to get the output sites back.
- Conditioning needs both `obs_times` and `obs_values`. Generating needs `predict_times`.
- `SDESimulator` does not condition on observations. Use a `Filter` to condition a continuous-time stochastic model.
- Run everything through the project uv environment with `uv run`.
