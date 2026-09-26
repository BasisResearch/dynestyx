# Discretization

Discrete-time inference methods need transitions between observation times, so
a continuous-time model has to be discretized before they can use it. There are
two ways to do this. Both take the same configurations and use the same
[automatic routing](#automatic-routing).

- **Effect-handler form**: wrap the model call in a `Discretizer` context. Your
  model code stays in continuous time and unchanged, so you can switch the
  discretization or the inference method by changing only the surrounding
  contexts. Use this with dynestyx's inference and simulation handlers, such as
  `Filter` or `LatentPathBuilder`.
- **Direct form**: call `discretize_dynamics` or
  `discretize_state_evolution` and get the discretized model or state evolution
  back as an object. Use this when you need the transition distributions
  themselves, for example to sample or score transitions in your own inference
  code.

## Effect-handler form

Place `Discretizer` *inside* the inference or simulation context, so the outer
handler receives the discretized dynamics:

```python
import dynestyx as dsx
from dynestyx.discretizers import (
    Discretizer,
    MeanTrajectoryLinearizationConfig,
)
from dynestyx.inference.filters import EnKFConfig, Filter

with Filter(EnKFConfig(n_particles=100)):
    with Discretizer(MeanTrajectoryLinearizationConfig()):
        result = model(obs_times=obs_times, obs_values=obs_values)
```

The configuration, here `MeanTrajectoryLinearizationConfig`, selects the
discretization method. See
[Discretizer configurations](../inference/configs/discretizer_configs.md) to
compare them.

## Direct form

`discretize_dynamics` returns a complete discrete-time model:

```python
import dynestyx as dsx
from dynestyx.discretizers import EulerMaruyamaConfig

discrete_dynamics = dsx.discretize_dynamics(
    continuous_dynamics,
    EulerMaruyamaConfig(),
)
```

The returned model keeps the initial condition, observation model, control
metadata and initial time; only the state evolution is replaced by the
discretized transition. To discretize a state evolution without building a
`DynamicalModel`, use [`discretize_state_evolution`](#state-evolution-only).

### Sampling and scoring

The returned model's distributions can be sampled and scored directly:

```python
initial_state = discrete_dynamics.initial_condition.sample(initial_key)
discrete_dynamics.initial_condition.log_prob(initial_state)

transition = discrete_dynamics.state_evolution(
    x=previous_state, u=control, t_now=t_now, t_next=t_next
)
state = transition.sample(key)
transition.log_prob(state)
```

`log_prob` is available when the configuration provides a transition density.
`EulerMaruyamaConfig` provides the density of its Gaussian approximation;
`DiffraxSampleConfig` only samples.

For dense all-pairs scores, [map over the transition distributions](../models/core/discrete_time_state_evolution.md#all-pairs-transition-scores).
For missing observations, use [masked_observation_log_prob](../models/core/observation_model.md#masked_observation_log_prob).

## Automatic routing

With no configuration, the method is chosen as follows:

- a deterministic ODE is integrated with `ODEFlowConfig()`, producing a Delta transition at the numerical flow endpoint;
- an `AffineDrift` with constant diffusion and no potential is discretized exactly; and
- other SDE models use Euler--Maruyama discretization by default.

Pass `ODEFlowConfig(simulator_config=ODESimulatorConfig(...), jitter_scale=...)` to customize ODE integration; all Diffrax settings are taken from the nested `ODESimulatorConfig`.

## Local affine-Gaussian parameters

Use [linearize_drift](../models/core/drifts.md#linearize_drift) to construct
a local affine drift, then discretize it with `ExactAffineConfig`. This
composition requires structurally constant additive diffusion:

```python
import dynestyx as dsx
from dynestyx.discretizers import ExactAffineConfig

cte = continuous_dynamics.state_evolution
local = dsx.StochasticContinuousTimeStateEvolution(
    drift=dsx.linearize_drift(
        cte.total_drift, x=reference_state, u=control, t=t_now
    ),
    diffusion=cte.diffusion,
)
transition = dsx.discretize_state_evolution(
    local, ExactAffineConfig(covariance_jitter=1e-8)
)
params = transition.params_at(t_now, t_next)

A = params.A
bias = params.bias
Q = params.cov
```

The resulting `LinearGaussianParams` give the approximation used by
`LocalLinearizationConfig` with matching covariance jitter. The drift is
linearized at `reference_state`, with time and control frozen at the left
endpoint. The fixed control can affect `A`, `bias`, and `cov`. The parameters
are conditional on that control, so `B=None`.

Use `vmap` to evaluate parameters along a reference trajectory. Trajectory
selection, observation linearization, and inference are handled separately.

For time-invariant affine drift with constant additive diffusion and no
potential term, discretization with `ExactAffineConfig(covariance_jitter=0)` produces a
`LinearGaussianStateEvolution` whose `params_at(t_now, t_next)` method returns
exact interval parameters up to floating-point error, assuming control is held
fixed over each interval. No linearization state is needed.

::: dynestyx.discretizers
    options:
      members:
        - discretize_dynamics
      show_root_heading: false
      show_root_toc_entry: false

## State evolution only

::: dynestyx.discretizers.discretize_state_evolution
