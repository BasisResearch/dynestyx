# LatentPathBuilder

Developer-facing reference for latent-path construction and scoring.

`LatentPathBuilder` exposes latent trajectories as NumPyro sample sites while
using pure array operations for state reconstruction and joint-density
evaluation.

## Package roles

- `dynestyx.inference.latent.builder`
  Interprets `dsx.sample(...)`, determines free coordinates, creates the latent
  sites, reconstructs once, scores once, and registers outputs. Plate iteration
  is a short loop here using shared slicing and stacking helpers.
- `dynestyx.inference.state_paths.reconstruct`
  Validates parameter shapes and reconstructs the full state path.
- `dynestyx.inference.state_paths.score`
  Returns the scalar `log p(x, y | ...)` for a fully assembled state path.
- `dynestyx.solvers`
  Owns continuous-time control-path evaluation plus the ODE and SDE trajectory
  wrappers shared by reconstruction and forward simulation.
- `dynestyx.observation_missingness`
  Owns concrete missing-coordinate metadata, completion, strategy resolution,
  and pure per-step observation scoring.
- `dynestyx.inference.utils.distribution_utils`
  Defines `_ForwardSimulationImproperUniform`: zero density with samples from
  dynamical forward simulation for initialization and forward execution.
- `dynestyx.inference.utils.plate_utils`
  Provides slicing, stacking, and NumPyro plate-frame suspension shared by the
  simulator and latent builder.
- `dynestyx.simulation.discrete` and `dynestyx.simulation.base`
  Provide the single-trajectory state sampler and conditional observation
  sampler reused for latent-site initialization.

## Execution path

The intended reviewer mental model is:

1. `_sample_ds(...)` slices each plate member when necessary and calls
   `_sample_single(...)`.
2. `_sample_single(...)` receives the observation views prepared by
   `dsx.sample(...)`, resolves concrete `MissingObservationMetadata`, and then
   chooses the ordinary, ODE, or exact-Dirac branch directly.
3. It creates `state_path_params` as a `_ForwardSimulationImproperUniform`. The
   distribution samples through the shared simulator state kernel and has zero
   log density.
4. It reconstructs the state path. If augmentation is active, it then creates a
   second zero-density improper site whose sampler calls the observation model
   conditional on that state path.
5. `compute_state_path_log_prob(...)` evaluates the complete scalar joint
   density once. The builder registers that value as the sole factor and emits
   the existing deterministic results.
6. Plate results are restacked, and any future-only rollout is forwarded to the
   surrounding simulator as before.

Because both latent distributions have zero density, no artificial Gaussian
terms are added and no base-log-probability corrections are subtracted.

## Replay metadata

Concrete calls compute `MissingObservationMetadata` from the observation mask.
Traced replay reuses compatible metadata scoped by site name, latent role,
observation shape, and requested strategy; replay without a compatible entry
raises an error explaining that a concrete model execution is required first.

## Public surface

The supported user-facing route is:

- `with dsx.LatentPathBuilder(): dsx.sample(...)` for explicit latent-path
  inference under NumPyro
- `dsx.log_prob(...)` for pure-JAX scoring of user-supplied latent paths

The reconstruction and scalar-scoring helpers remain available as
developer-facing building blocks.

::: dynestyx.inference.latent.builder
    options:
      filters: []
