# Discretization

A continuous-time `DynamicalModel` can be converted into a discrete-time model
without entering an effect-handler context:

```python
import dynestyx as dsx
from dynestyx.discretizers import EulerMaruyamaConfig

discrete_dynamics = dsx.discretize_dynamics(
    continuous_dynamics,
    EulerMaruyamaConfig(),
)
```

The returned model preserves the initial condition, observation model, control
metadata, and initial time. Its state evolution is the interval transition
selected by the discretizer configuration.

## Sampling and scoring

Use the returned model's distributions directly:

```python
initial_state = discrete_dynamics.initial_condition.sample(initial_key)
discrete_dynamics.initial_condition.log_prob(initial_state)

transition = discrete_dynamics.state_evolution(
    x=previous_state, u=control, t_now=t_now, t_next=t_next
)
state = transition.sample(key)
transition.log_prob(state)
```

Sampling requires a transition with `sample`; scoring additionally requires
`log_prob`. `EulerMaruyamaConfig` supplies both for its Gaussian approximation.
`DiffraxSampleConfig` supplies sampling only.

For dense all-pairs scores, [map over the transition distributions](../models/core/discrete_time_state_evolution.md#all-pairs-transition-scores).
For missing observations, use [masked_observation_log_prob](../models/core/observation_model.md#masked_observation_log_prob).

## Effect-handler form

A `Discretizer` maps a `ContinuousTimeStateEvolution` to a `DiscreteTimeStateEvolution` by discretizing the corresponding ODE or SDE; the resulting model is compatible with discrete-time inference techniques in `dynestyx` when the selected transition interface supplies what the inference method requires. The discretizer context should be placed *inside* the corresponding inference context:

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

The config (in the above, `MeanTrajectoryLinearizationConfig`) changes the corresponding method for discretizing the continuous-time dynamics. See [Discretizer configurations](../inference/configs/discretizer_configs.md) for more information about each.

## Automatic routing

When no configuration is supplied, `Discretizer()` chooses automatically:

- a deterministic ODE is integrated with `ODEFlowConfig()`, producing a Delta transition at the numerical flow endpoint;
- an `AffineDrift` with constant diffusion and no potential is discretized exactly; and
- other SDE models use Euler--Maruyama discretization by default.

Pass `ODEFlowConfig(simulator_config=ODESimulatorConfig(...), jitter_scale=...)` to customize ODE integration; all Diffrax settings are taken from the nested `ODESimulatorConfig`.

Both `discretize_dynamics()` and `Discretizer()` use this routing.

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

Use `discretize_state_evolution` to discretize a continuous state evolution
without constructing a `DynamicalModel`.

::: dynestyx.discretizers.discretize_state_evolution
