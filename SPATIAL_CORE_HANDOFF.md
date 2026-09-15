# Spatial dynamics core handoff

## Status

- Branch: `md-spatial`
- Base: `origin/main` at `1c5d89f` (`Bump to cd-dynamax v0.4.3 and dynestyx 0.5.1 (#362)`)
- Purpose: isolate the reusable spatial-state machinery from the larger
  `md-pde-improvements` branch.
- Focused verification: `39 passed` in `tests/test_spatial_state_evolution.py`.
- Repository lint/type checks: `ty`, `ruff check`, and `ruff format --diff` all pass.

This branch is intentionally limited to the spatial dynamics core. It does not contain
PDE notebooks, Navier--Stokes benchmarks, Exponax dependency changes, grid interpolation,
or spatial downsampling.

## What changed

### `dynestyx/models/spatial.py`

This new module adapts component-first spatial fields to Dynestyx's flat state-vector
contract.

Public objects:

- `FieldLayout`: records one or more field shapes and provides a bijection between fields
  and a concatenated flat state. `flatten` and `unflatten` preserve arbitrary leading batch
  axes.
- `SpatialStateEvolution`: a `DiscreteTimeStateEvolution` backed by a fixed-step spatial
  solver. It applies the solver `n_substeps` times with `jax.lax.scan`, checks that model
  time increments match `dt_obs`, optionally threads an auxiliary carry within each
  transition, and optionally adds diagonal process noise after the full transition.
- `field_observation`: constructs either an exact identity observation or a diagonal
  Gaussian identity observation for a declared field layout.
- `spatial_dynamics`: convenience factory assembling an initial distribution,
  `SpatialStateEvolution`, and `field_observation` into a `DynamicalModel`.

Field shapes follow `(n_components, *spatial_shape)`. A single field is specified as, for
example, `(3, 64, 64, 64)`; coupled fields are specified as a sequence such as
`((2, 64, 64), (1, 128, 128))`.

Noise specifications accept a scalar, a value per field, or an array broadcastable to a
single field. Noise is represented as scalar/vector standard deviations, never as a dense
state-sized covariance. `None` selects a `Delta`; non-`None` process noise returns
`Normal(...).to_event(1)`.

The stepper is a dynamic Equinox field so array-valued solver parameters remain visible as
pytree leaves. Construction uses `jax.eval_shape` to validate solver output shapes without
performing solver work. Validation can be disabled for steppers that cannot be abstractly
traced, including some host callbacks.

### `dynestyx/models/observations.py`

Adds `DiagonalGaussianObservation`. It stores scalar or vector standard deviations and an
optional measurement function instead of constructing a dense covariance. The spatial core
uses the identity form for noisy field observations. The optional measurement-function form
is generic and is retained because it is part of the class's tested contract.

### Package exports

`dynestyx/__init__.py` and `dynestyx/models/__init__.py` export:

- `DiagonalGaussianObservation`
- `FieldLayout`
- `SpatialStateEvolution`
- `field_observation`
- `spatial_dynamics`

### Tests

`tests/test_spatial_state_evolution.py` covers:

- single- and multi-field layout validation and round trips;
- preservation of batch axes;
- substep execution and time-step validation;
- abstract output-shape validation without solver execution;
- dynamic pytree leaves for steppers and noise;
- auxiliary carry semantics;
- simulation equivalence with a hand-written rollout;
- `jax.pure_callback` integration for external solvers;
- deterministic and diagonal-Gaussian process/observation models;
- scalar, per-field, and per-component noise broadcasting;
- simulator, log-probability, EKF, and EnKF smoke coverage; and
- Gaussian initial-condition support through `initial_std`.

## Important semantics and limitations

- `n_substeps` solver applications make one model transition. The caller must ensure the
  solver's internal step size times `n_substeps` equals `dt_obs`.
- Process noise is added once per model transition, after all solver substeps. It is white
  and is not scaled automatically by `dt_obs`.
- Auxiliary carry is initialized at every model-transition boundary and only persists
  between substeps within that transition. Persistent physical quantities belong in the
  state itself.
- A host-side solver can be wrapped with `jax.pure_callback`, but it is not differentiable.
  Such a model cannot support EKF linearization or gradient-based inference through the
  solver.
- The diagonal `Normal` distributions intentionally do not expose a dense
  `covariance_matrix`. Simulator, ensemble-filter, particle-filter, and log-probability
  paths are appropriate; covariance-reading Kalman backends are not.
- `spatial_dynamics` defaults to deterministic initial state, transition, and identity
  observation. For EKF use, set `initial_std`, `process_noise_std`, and
  `observation_noise_std`; a `Delta` initial condition leads to a degenerate covariance.
- Full-field identity observations duplicate the state trajectory in simulation and can be
  prohibitively large for filtering. Observation operators and `FieldDownsampler` are
  deliberately deferred to a later PR.

## Deliberately excluded work

The source branch also contains the following, none of which is on `md-spatial`:

- `FieldDownsampler`, `DiracObservation`, and operator-aware extensions to
  `field_observation`/`spatial_dynamics`;
- `dynestyx/observation.py` and `GridInterpolator`;
- Kuramoto--Sivashinsky and Navier--Stokes notebooks;
- `ns_performance.py` and generated benchmark results;
- the `pdes`/Exponax dependency group; and
- notebook media, metadata changes, and `.gitignore` changes.

The natural follow-up stack is: generic/non-identity observation support, then
`FieldDownsampler`, then tutorials and benchmarks.

## Verification commands

Run the focused tests with:

```bash
DYNESTYX_PF_PARTICLES_SCALE=0.01 uv run pytest tests/test_spatial_state_evolution.py -q
```

Repository convention reserves the full suite for the user. Before merging, the user can
run:

```bash
uv run scripts/clean.sh
uv run scripts/lint.sh
uv run scripts/test.sh
```

## Extraction notes

The core implementation originated in commit `70c6e06` on top of `pde_v2`. The relevant
five-file patch was applied onto current `origin/main` rather than carrying over the old
branch history. Later commit `a5e40d8` only adds downsampling/operator integration within
these files; it makes no corrections to the pre-downsampling spatial core, so those changes
were intentionally omitted.
