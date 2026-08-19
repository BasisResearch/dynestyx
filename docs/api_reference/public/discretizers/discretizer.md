# Discretizers


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
