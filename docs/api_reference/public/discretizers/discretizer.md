# Discretizers


A `Discretizer` maps a `ContinuousTimeStateEvolution` to a `DiscreteTimeStateEvolution` by discretizing the corresponding SDE; the resulting model is compatible with discrete-time inference techniques in `dynestyx` (i.e., discrete time filters, smoothers, and `LatentPathBuilder`). The discretizer context should be placed *inside* the corresponding inference context:

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

The config (in the above, `MeanTrajectoryLinearizationConfig`) changes the corresponding method for discretizing the SDE. See [Discretizer configurations](../inference/configs/discretizer_configs.md) for more information about each.

## Automatic routing

When no configuration is supplied, `Discretizer()` chooses automatically:

- an `AffineDrift` with constant diffusion and no potential is discretized exactly; and
- other models use Euler-Maruyma discretization by default.