# LatentPathBuilder

Developer-facing reference for latent-path construction and scoring.

The latent-path refactor now separates responsibilities across two layers:

- `dynestyx.inference.latent`: NumPyro-facing orchestration
- `dynestyx.inference.state_paths`: pure-JAX state-path layout, reconstruction,
  and scoring

## Package roles

- `dynestyx.inference.latent.builder`
  `LatentPathBuilder` itself. This is the handler that interprets
  `dsx.sample(...)` in NumPyro mode, creates the dummy latent sites, then
  registers factors and deterministic outputs.
- `dynestyx.inference.latent.prepare`
  Request canonicalization for one latent-path call. This is where the builder
  resolves the concrete latent layout and shape-only example arrays needed for
  site creation.
- `dynestyx.inference.latent.plate`
  Plate-specific splitting and restacking logic for hierarchical latent-path
  inference.
- `dynestyx.inference.state_paths.layout`
  Structural planning for `state_path_params`, optional
  `missing_obs_values`, and the reconstruction rule `x = g(z)`.
- `dynestyx.inference.state_paths.reconstruct`
  Pure-JAX reconstruction of the full state path once concrete latent values
  are known.
- `dynestyx.inference.state_paths.score`
  Pure-JAX scoring of `log p(x, y | ...)` from a fully assembled state path.

## Execution path

The intended reviewer mental model is:

1. `LatentPathBuilder._sample_ds(...)` routes one trajectory or plate member to
   `_sample_single(...)`.
2. `_prepare_latent_path_request(...)` resolves the concrete layout from
   `dynamics`, `obs_times`, and `obs_values`, and canonicalizes any provided
   latent values.
3. `LatentPathBuilder` creates dummy NumPyro sites for `state_path_params` and,
   when needed, `missing_obs_values`.
4. The builder reconstructs the full state path in pure JAX through
   `prepared.layout.assemble_from_params(...)`.
5. The builder scores that assembled path in pure JAX through
   `compute_state_path_log_prob_terms(...)`.
6. Only after the pure-JAX work is done does the builder attach NumPyro
   `factor(...)` and `deterministic(...)` sites for the realized outputs.

This is the key design point of the refactor: NumPyro owns site registration,
but latent-path semantics live in pure JAX.

## Public surface

The supported user-facing route is:

- `with dsx.LatentPathBuilder(): dsx.sample(...)` for explicit latent-path
  inference under NumPyro
- `dsx.log_prob(...)` for pure-JAX scoring of user-supplied latent paths

Lower-level helpers such as `prepare_latent_path_layout(...)` are still useful
for tests and implementation work, but they should be treated as
developer-oriented machinery rather than the primary public workflow.

::: dynestyx.inference.latent.builder
    options:
      filters: []
