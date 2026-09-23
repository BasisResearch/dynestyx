Dynestyx represents states and observations as scalars or one-dimensional arrays. Many
dynamical systems live on more complex data structures, for example:

- tensors of shape `(d_1, d_2, ..., d_n)`;
- tuples of tensors `((c_1, c_2, ..., c_n), (d_1, d_2, ..., d_n), ...)`; and
- general pytrees.

## Summary of the proposal

Allow users to define dynamics on fixed pytrees (see the [JAX pytrees documentation](https://docs.jax.dev/en/latest/101/pytrees.html)) and provide a utility for communicating easily with Dynestyx primitives.

**Desiderata:**

1. Keep the underlying Dynestyx internal operations intact: they should act on vectors,
   which is necessary for data assimilation.
2. Preserve the current API for dynamics defined on vectors.
3. Allow users to define their dynamics using their chosen data structure.
4. Return results using the chosen data structure.

## Construction in `DiscreteTimeStateEvolution`

There are three ingredients:

1. `initial_condition`: a NumPyro distribution over the initial state.
2. `state_evolution`: a discrete-time callable
   `(x, u, t_now, t_next) -> numpyro.distributions.Distribution`.
3. `observation_model`: a callable
   `(x, u, t) -> numpyro.distributions.Distribution`.

**Difficulty:** There does not seem to be a canonical way to define a `Distribution` over
pytrees, only over tensors.

The proposed `Layout` utility translates between the structured space and the flat space.

Example API:

```python
state_layout = Layout.from_example(x)
x_flat = state_layout.flatten(x)  # Produces a vector.
x = state_layout.unflatten(x_flat)  # Produces a pytree.
state_dim = state_layout.dim  # Returns the dimension of flat x.
```

The layout can then be passed to the dynamics definition:

```python
dynamics = DynamicalModel(
    initial_condition=initial_condition,
    state_evolution=evolution,
    observation_model=observation_model,
    state_dim=state_layout,  # This could also have its own dedicated keyword.
    observation_dim=observation_layout,
    control_dim=control_layout,
)
```

### Option 1

When a layout is provided, it creates the following contract: a user-specified callable
always receives structured data but must return a `Distribution` over the flat space.

Example API:

```python
def state_evolution(x, u, t_now, t_next):
    x_next = F(x, u)  # User state transition defined over the structured space.
    x_next = state_layout.flatten(x_next)
    x_distribution = dist.Normal(loc=x_next, scale=...)
    return x_distribution


def observation_model(x, u, t):
    y = g(x, u)  # User-defined observation defined over the structured space.
    y = observation_layout.flatten(y)
    y_distribution = ...
    return y_distribution
```

### Option 2

The user receives flat data, and the utilities are provided to help reshape it. Dynestyx
does not reshape anything internally. In this case, passing a layout to the dynamics is
required only to reshape results. Is this necessary or desirable?

Example API:

```python
def state_evolution(x, u, t_now, t_next):
    x = state_layout.unflatten(x)  # Manual user transformation.
    u = control_layout.unflatten(u)
    x_next = F(x, u)  # User state transition defined over the structured space.
    x_next = state_layout.flatten(x_next)
    x_distribution = dist.Normal(loc=x_next, scale=...)
    return x_distribution


def observation_model(x, u, t):
    x = state_layout.unflatten(x)  # Manual user transformation.
    u = control_layout.unflatten(u)
    y = g(x, u)  # User-defined observation defined over the structured space.
    y = observation_layout.flatten(y)
    y_distribution = ...
    return y_distribution
```

## `SimulatedResult`

`SimulatedResult` can receive the layouts and automatically reshape its data into the
structured spaces.

```python
return SimulatedResult(
    state_layout=dynamics.state_layout,
    observation_layout=dynamics.observation_layout,
    times=_tile_times(times, n_sim),
    x_0=initial_state,
    states=_ensure_trailing_dim(states),
    observations=_ensure_trailing_dim(observations),
).unflatten()

# Data are automatically reshaped internally.
result = dsx.simulate(
    dynamics,
    rng_key=jr.key(13),
    predict_times=times,
)
```

## Policies

Under the existing contract, policies must receive a `Distribution` object defined over
the flat space.

```python
class Policy(eqx.Module):
    def __call__(self, x_hat, t_now, t_next, s):
        x = x_hat.mean  # This is the mean in the flat space.
        x = state_layout.unflatten(x)  # Move to the structured space.
        u = g(x, s)  # Policy defined over the structured space.
        return u  # Must be shaped according to control_layout.
```

### Helper functions for initial conditions, state transitions, and observations

For frequently used distributions, we can provide or expand existing utilities such as
`GaussianStateEvolution` that take in callable and produce valid `Distributions`. 

```python
# Gaussian state evolution.
state_transition = GaussianStateEvolution(
    F=F,
    state_layout=state_layout,
    cov=sigma,
)

# Deterministic state evolution.
state_transition = DiracStateEvolution(
    F=F,
    state_layout=state_layout,
)
```

Internally, both do similar things:

```python
class DiracStateEvolution(DiscreteTimeStateEvolution):
    """Deterministic discrete transition."""

    F: Callable

    def __init__(self, F: Callable, *, state_layout: Layout | None = None):
        self.F = F
        self.state_layout = state_layout

    def mean(self, x, u, t_now, t_next):
        return self._flatten_state(self.F(x, u, t_now, t_next))
        # Alternatively, using Option 2:
        # x = self._unflatten_state(x)
        # u = self._unflatten_control(u)
        # return self._flatten_state(self.F(x, u, t_now, t_next))

    def __call__(self, x, u, t_now, t_next):
        loc = jnp.asarray(self.mean(x, u, t_now, t_next))
        return dist.Delta(loc, event_dim=0 if loc.ndim == 0 else 1)
```

`cov` can be treated according to different conventions:
- `scalar`: i.i.d. across flat state dimensions;
- `vector` or pytree-shaped: independent Gaussians with the given variance;
- `matrix`: covariance in the flat space; and
- `callable`: maps `(x, u, t_now, t_next) -> cov`, where `cov` is the covariance in
  flat space.

We can do similar things for initial conditions and observations:

```python
initial_condition = DiracInitialCondition(x_0, state_layout=state_layout)
evolution = DiracStateEvolution(F=step, state_layout=state_layout)
observation_model = DiracObservation(
    h=observation_function,
    state_layout=state_layout,
    observation_layout=observation_layout,
)

initial_condition = GaussianInitialCondition(
    x_0,
    state_layout=state_layout,
    cov=cov,
)
evolution = GaussianStateEvolution(
    F=step,
    state_layout=state_layout,
    cov=cov,
)
observation_model = GaussianObservation(
    h=observation_function,
    state_layout=state_layout,
    observation_layout=observation_layout,
    cov=cov,
)
```
