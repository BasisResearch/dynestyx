# Overview

Simulators run pure-JAX forward trajectories for a
`DynamicalModel` on a chosen time grid, then optionally attach the realized
outputs to a NumPyro trace when used through `dsx.sample(...)`.

For pure-JAX forward generation, user-facing code should prefer
`dsx.simulate(...)`. The simulator classes documented here are the NumPyro-aware
handler implementations used by `dsx.sample(...)` and by posterior rollout from
`Filter`/`Smoother`.

!!! note "When to use each time argument"
    - **`predict_times`**: use this when you want rollout trajectories at specific
      times for simulation and/or post-filter rollout.
      - In posterior-rollout mode, predictions are generated at `predict_times`
        from inference-handler posteriors.
      - Typical use: forward simulation, forecasting, or dense trajectories for
        visualization.
    - **`obs_times` / `obs_values`** are consumed by observation-aware handlers
      such as `LatentPathBuilder`, `Filter`, and `Smoother`, not by the public
      simulator interface itself.
    - **If `predict_times` is omitted**: the simulator does not run and adds no
      deterministic sites.

!!! note "Context and caveats"
    - **NumPyro context required for `dsx.sample(...)`**: simulator handlers draw
      randomness from the active NumPyro PRNG key, but the rollout itself is pure
      JAX and sites are registered only at the end. Use `dsx.simulate(...)` for
      the explicit pure-JAX API.
    - **Generation-only public API**: raw `Simulator`, `DiscreteTimeSimulator`,
      `ODESimulator`, and `SDESimulator` calls expect `predict_times`, not direct
      observation conditioning.
    - **Inference lives elsewhere**: use `LatentPathBuilder` for explicit latent
      paths, `Filter` for marginalized inference, and `Smoother` for smoothing.
      Simulators can then wrap those handlers for rollout with `predict_times`.

!!! note "Deterministic sites"
    When simulator trajectories are produced, sites are recorded as `"{name}_{key}"`
    where `name` is the first
    argument to `dsx.sample(name, dynamics, ...)` (conventionally `"f"`):

    - `"f_x_0"`: realized initial-state draw, shape `(n_sim, state_dim)`,
    - `"f_times"`: trajectory time grid, shape `(n_sim, T)`,
    - `"f_states"`: latent trajectory, shape `(n_sim, T, state_dim)`,
    - `"f_observations"`: sampled observations, shape `(n_sim, T, obs_dim)`.

    In filter-rollout mode (`predict_times` with filtered posteriors), additional
    keys `"f_predicted_states"`, `"f_predicted_times"`, and
    `"f_predicted_observations"` are recorded. Segment-level rollouts also
    register realized anchor-state sites such as `"f_1_x_0"` when applicable.

    Under `numpyro.infer.Predictive(model, num_samples=N)`, NumPyro prepends a leading
    `num_samples` axis, giving final shapes `(num_samples, n_sim, T, dim)`.
    Use `dynestyx.flatten_draws` to collapse the `(num_samples, n_sim)` prefix into one
    axis for plotting or downstream analysis.

    If `predict_times` is omitted, no public simulator rollout is performed and
    these sites are not added.

## Simulators

::: dynestyx.simulation
    options:
      filters: []
      show_root_heading: false
      show_root_toc_entry: true
  
