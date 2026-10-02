# LatentPathBuilder

One way to do Bayesian inference on a dynamical system is to "unroll" it and perform joint inference. For example, consider a dynamical system 

$$
x_{t+1} \sim p(x_{t+1} \,|\, x_t)
$$ 

with observations

$$
y_t \sim p(y_t \,|\, x_t).
$$

Then we can perform inference on the joint distribution $p(x_{0:T}, \theta \,|\, y_{1:T})$ by considering a probabilistic program that computes recursions in a `for` loop, conceptually something like
```python
def model(ys):
  x_0 = numpyro.sample("x0", p_x0)
  theta = numpyro.sample("theta", p_theta)

  x_i = x_0

  for i in range(T):
    x_i = numpyro.sample(f"x_{i+1}", p_transition(x_i))
    numpyro.sample(f"y_{i+1}", p_observation(x_i), obs=ys[i])
```

`LatentPathBuilder` provides such a representation for joint learning, with support for arbitrary missingness in the data `y`. To support missingness and be more efficient, the pseudocode above *is not* how `LatentPathBuilder` is implemented, but is a useful mental model.

`LatentPathBuilder` is a NumPyro-facing handler: use it through `dsx.sample(...)` inside a
NumPyro model. For pure-JAX trajectory scoring of fixed latent values, use
`dsx.log_prob(...)` instead.

Conceptually:

- `Simulator` generates trajectories and observations.
- `LatentPathBuilder` constructs `state_path_params`, reconstructs the full
  `state_path`, and evaluates `log p(x, y | ...)`.
- `Filter` and `Smoother` remain the preferred handlers when observations should be marginalized.

The main output names are intentionally distinct from simulator rollout names:

- `f_state_path` / `f_state_path_times`: explicit latent-path inference outputs
  from `LatentPathBuilder`
- `f_states` / `f_times`: rollout outputs from `Simulator` and `dsx.simulate(...)`

## JIT and the builder cache

The shapes of NumPyro sites constructed by `LatentPathBuilder` depend on the
missingness pattern. They must therefore be known at compile time.

There are three paths:

1. Run `LatentPathBuilder` concretely to compute the missingness.
2. Run it concretely once to fill its cache, and then run it under JIT. NumPyro
   MCMC routines typically do this automatically.
3. Provide the missingness metadata directly.

Path 2 must use the same `LatentPathBuilder` object because that object stores
the cache. The recommended pattern is to define the model first and apply the
handler when the model runs. Examples of the three paths follow.

### 1. Concrete execution

```python
def conditioned_model(obs_times=None, obs_values=None):
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )

predictive = Predictive(conditioned_model, num_samples=10)
with dsx.LatentPathBuilder():
    result = predictive(
        prediction_key,
        obs_times=obs_times,
        obs_values=obs_values,
    )
```

### 2. Cached metadata

```python
builder = dsx.LatentPathBuilder()
with builder:
    predictive(
        warmup_key,
        obs_times=obs_times,
        obs_values=obs_values,
    )
    result = jax.jit(predictive)(
        prediction_key,
        obs_times=obs_times,
        obs_values=obs_values,
    )
```

### 3. Explicit metadata

`dsx.prepare_missing_observation_metadata(...)` can create the metadata from
concrete data. You can also create it directly:

```python
obs_times = jnp.array([0.0, 1.0])
obs_values = jnp.array([[0.0, jnp.nan], [jnp.nan, 1.0]])

metadata = dsx.MissingObservationMetadata(
    missing_obs_times=jnp.array([0.0, 1.0]),
    missing_obs_coordinate_indices=jnp.array([1, 0], dtype=jnp.int32),
    missing_flat_indices=jnp.array([1, 2], dtype=jnp.int32),
    observation_shape=(2, 2),
    has_missing=True,
    has_partial_missing=True,
    has_fully_missing_rows=False,
)
```

All fields must match the layout of `obs_values`.

```python
def conditioned_model(obs_times=None, obs_values=None):
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        missing_obs_metadata=metadata,
    )

predictive = jax.jit(Predictive(conditioned_model, num_samples=10))
with dsx.LatentPathBuilder():
    result = predictive(
        prediction_key,
        obs_times=obs_times,
        obs_values=obs_values,
    )
```

Observation times and finite values can change. The missing-observation layout
must agree with the cached or supplied layout. If you supply one metadata
object, all plate members use its layout.

For plate members with different layouts, use one builder for all traced calls.
Ragged `LatentStateResult` fields are flat lists of per-member arrays. Each
NumPyro site keeps its shape, and the rightmost plate index varies fastest.

## Implementation Details

To support arbitrary missingness and improve efficiency, the implementation of `LatentPathBuilder` differs from the simple "unrolling" mental model. For deterministic ODEs, `f_state_path_params` is a proper NumPyro sample site backed by the model's initial-condition distribution. For other paths, the state path is initialized through a modified improper uniform distribution whose `sample` method draws from the state-space-model prior. In particular:

- discrete `state_path_params` are drawn from the state-space-model prior;
- ODE `state_path_params` are drawn from the initial-condition distribution and retain a singleton leading time axis;
- exact-observation parameters retain only the free state coordinates; and
- remaining `missing_obs_values` are drawn from the observation model given the simulated state path.

After simulation, the model is scored according to the joint log density $\log p(x, y | \theta)$. For deterministic ODEs, the `f_state_path_params` site contributes the initial-condition density, so `f_joint_log_prob_factor` contributes only the remaining terms. For other paths, the complete joint density is registered through `f_joint_log_prob_factor`. A transformed ODE initial condition can use NumPyro's `TransformReparam` with `config={"f_state_path_params": TransformReparam()}`.

::: dynestyx.inference.latent.builder
    options:
      members:
        - LatentPathBuilder
