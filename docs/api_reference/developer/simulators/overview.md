# Overview

Simulators (also called *unrollers*) generate forward trajectories from a
`DynamicalModel` on a provided time grid, and can also sit outside inference
handlers to produce posterior rollouts.

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
      JAX and the realized sites are registered only at the end. Use
      `dsx.simulate(...)` when you want a pure-JAX API with an explicit
      `rng_key`.
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

User code will usually choose between:

- `with Simulator(): ...` or a concrete simulator handler inside a NumPyro model
- `dsx.simulate(...)` for pure-JAX forward simulation without NumPyro sites

## BaseSimulator

::: dynestyx.simulation.base.BaseSimulator
    options:
      show_root_heading: false
      show_root_toc_entry: false
