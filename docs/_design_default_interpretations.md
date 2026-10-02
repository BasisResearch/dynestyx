# Default model interpretations

Proposed defaults for `dsx.sample()`, based on the model, data arguments, and
handler stack. Explicit handlers and configs take precedence.

## Simulation

In all cases, the default `Discretizer` is applied for continuous time models.

- `predict_times`, no simulator: add `Simulator()` on the outside.
  - This rule also applies if `ctrl_times` and `ctrl_values` are both supplied 

All other combinations of arguments are treated as ambiguous and raise an error.

## Learning

When `obs_times` and `obs_values` are supplied and no learning handler (Filter/Smoother/LatentPathBuilder) is specified:


| Model                                                                     | Default                      |
| ------------------------------------------------------------------------- | ---------------------------- |
| `obs_dim != state_dim`                                                    | Filter using the order below |
| `obs_dim == state_dim`, discrete-time with an explicit transition density | Size-based choice below      |
| `obs_dim == state_dim`, ODE                                               | Size-based choice below      |
| `obs_dim == state_dim`, SDE                                               | Filter using the order below |

For the size-based choice, use `T * obs_dim`, where `T` is the number of
observation times: below a cutoff, choose `dsx.LatentPathBuilder` (LPB); at or
above it, choose a Filter using the order below. The cutoff is TBD and will be
determined later through simple laptop experiments run by Codex.

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
