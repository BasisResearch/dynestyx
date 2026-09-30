# Default model interpretations

Proposed defaults for `dsx.sample()`, based on the model, data arguments, and
handler stack. Explicit handlers and configs take precedence.

## Simulation

- `predict_times`, no simulator: add `Simulator()` on the outside.
- `predict_times` + `ctrl_times`, no `ctrl_values`: use a closed-loop simulator
instead. A control policy must be supplied. For continuous-time models, apply
the default `Discretizer()` first.

## Learning

When `obs_times` and `obs_values` are supplied and no learning handler (Filter/Smoother/LatentPathBuilder) is specified:


| Model                                                                     | Default                      |
| ------------------------------------------------------------------------- | ---------------------------- |
| `obs_dim != state_dim`                                                    | Filter using the order below |
| `obs_dim == state_dim`, discrete-time with an explicit transition density | `dsx.LatentPathBuilder`      |
| `obs_dim == state_dim`, ODE                                               | `dsx.LatentPathBuilder`      |
| `obs_dim == state_dim`, SDE                                               | Filter using the order below |


Choose the first compatible filter:

1. **KF** using `cuthbert` with `Discretizer()` (default Discretizer will recognize linearity if possible).
2. **EnKF**; for continuous-time models, use `DiffraxSampleConfig`. (Applicable only for Gaussian observations.)
3. **PF**; for continuous-time models, use `DiffraxSampleConfig`.

Continuous time models are discretized here by default due to the better support of missing observations in `cuthbert`.

If observations and prediction times are both supplied, apply both the learning
and simulation rules.

## To explore

Implementation of the above may lead us to examining the handler stack in full; in this case, we should also provide informative errors due to common mis-orderings of the handler stack (e.g., putting `Discretizer` outside of the inference handler).

Auto-selecting `DiracIdentityObservation` + `LatentPathBuilder` based on data
smoothness and sampling density along a smoothed interpolant. Criteria are TBD;
equal observation and state dimensions alone do not imply identity observations.