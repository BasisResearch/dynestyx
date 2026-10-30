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

## Default interpretations

`sample` and `condition` supply missing inference and prediction handlers, and
issue a standard Python warning naming the choices. Explicit handler contexts
override these choices; supplying a handler also lets you configure its algorithm.

| Inputs | Automatic behavior |
| --- | --- |
| `predict_times` | Add `Simulator()` if none is active |
| `obs_times` and `obs_values` | Add an inference handler if none is active |
| Observations and prediction times | Apply both rules |
| An incomplete observation or control pair | Raise an input-validation error |

Categorical latent states select exact HMM filtering with `HMMConfig()`.
For other models, `sample` selects `LatentPathBuilder()` when observation and state
dimensions match, `T * observation_dim < 1,000`, and the transition is known to
have a density or the model is a native ODE. This initial size heuristic uses
the number of observation times `T` per trajectory,
without multiplying by plate sizes. An ODE remains native: LPB samples its initial state and integrates the
path. Unknown custom transition capabilities use filtering; request LPB explicitly
if your custom transition supports it.

Other cases choose the first structurally compatible filter: cuthbert Kalman
filter, ensemble Kalman filter for supported Gaussian observation models, then
bootstrap particle filter. Automatic particle filtering rejects
`DeterministicObservation`, including `DiracIdentityObservation`; add observation
noise explicitly with `GaussianRelaxation(observation_model_cov=...)`.
`condition` always takes this filter route, as does
an active `Evaluation`, which requires filtered results. For automatic Evaluation,
the compatible choice is cuthbert EnKF for supported Gaussian observations:
cuthbert KF/PF do not yet supply canonical predictive-observation diagnostics.
HMM filtering does not supply these outputs, so automatic Evaluation rejects
categorical latent states. Other observation models require an explicit compatible
filter. `condition` does not register NumPyro sites; stochastic algorithms retain
their existing seed/key requirements. Use an explicit filter config with `crn_seed` where needed, or
`dsx.simulate(..., rng_key=...)` for pure-JAX simulation.

Without observations, automatic simulation preserves the model's native
representation: `Simulator()` selects `SDESimulator` for SDEs and `ODESimulator`
for ODEs. Their internal integration steps are independent of `predict_times`.
An explicit `Discretizer` instead selects simulation of the discrete transitions.
Automatic continuous-time filtering adds discretization: Kalman filtering uses
the default discretizer, nonlinear SDE ensemble/particle filtering uses
`DiffraxSampleConfig`, and ODE filtering uses `ODEFlowConfig`.
Explicit inference handlers retain their continuous-time behavior, including when
an automatic simulator is added. An explicit native simulator or solver config
must remain compatible with the model reaching it; incompatible combinations
raise instead of changing that configuration.

Existing limitations still apply: LPB predictions must be at or after the final
observation time, and LPB with traced observation arrays needs fixed missingness
metadata supplied through `missing_obs_metadata`. No default infers identity
observations or adds numerical jitter. Equal state and observation dimensions
alone do not imply noise-free observations.

For example, a model can request predictions without an explicit simulator:

```python
with numpyro.handlers.seed(rng_seed=0):
    prior = dsx.sample("f", dynamics, predict_times=times)

with numpyro.handlers.seed(rng_seed=1), Filter(KFConfig(filter_source="cuthbert")):
    posterior = dsx.sample(
        "f", discrete_dynamics,
        obs_times=times, obs_values=observations,
        predict_times=future_times,
    )
```

## Handler order

Nest handlers in this order, **outermost first**:

1. `Evaluation`
2. `Simulator`
3. `Filter`, `Smoother`, or `LatentPathBuilder`
4. [GaussianRelaxation](relaxations/gaussian_relaxation.md)
5. [Discretizer](discretizers/discretizer.md)
6. `plate`

```python
with dsx.Filter(filter_config):  # outermost
    with dsx.GaussianRelaxation(observation_model_cov=0.1):
        with dsx.Discretizer(discretizer):
            result = dsx.condition(
                "trajectory",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )
```

Outermost means the least-indented `with` block. Here, `Filter` surrounds
`GaussianRelaxation`, which surrounds `Discretizer`. The model is discretized,
then relaxed, then passed to the filter. Optional stages are omitted.

Omit stages you do not need. Use one handler per stage, choosing one of the
three inference handlers; nested plates may repeat. In a single `with`
statement, the leftmost context is outermost.

Model processing proceeds from the inside out: plates are handled first, then
discretization and relaxation prepare the model for inference, followed by
simulation and evaluation. This lets an outer inference handler receive the
transformed model.

`sample` and `condition` supply missing inference and simulation stages as
described above. Direct calls to `simulate` and `log_prob`
use the model passed to them; apply model transformations explicitly with
`discretize_dynamics` or `relax_dynamics` first when needed.

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
