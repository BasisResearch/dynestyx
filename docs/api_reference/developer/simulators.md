# Overview

Simulators (also called *unrollers*) turn a `DynamicalModel` into explicit NumPyro
`sample` sites for latent states and observations on a chosen time grid.

!!! note "When to use each time argument"
    - **`obs_times` and `obs_values` must be provided together**:
      - `obs_times` defines where observation sample sites (`y_t`) live.
      - `obs_values` provides conditioning values for those sites via `obs=...`.
      - Typical use: observed-data simulation/inference on a known observation grid.
    - **`predict_times`**: use this when you want rollout trajectories at specific
      times for simulation and/or post-filter rollout.
      - In filter-rollout mode, predictions are generated at `predict_times` from
        filtered posteriors.
      - Typical use: forward simulation, forecasting, or dense trajectories for
        visualization.
    - **If both are provided**:
      - `obs_times` controls filtering/conditioning points.
      - `predict_times` controls where predicted trajectories are reported.
    - **If both are omitted**: simulator does not run and adds no deterministic sites.

!!! note "Context and caveats"
    - **NumPyro context required**: simulators call `numpyro.sample(...)` and draw
      randomness via NumPyro PRNG keys, so they must run inside a NumPyro model
      (or a `numpyro.handlers.seed(...)` context).
    - **Conditioning is optional**: if `obs_values` is provided (e.g.
      `dsx.sample(..., DynamicalModel(...), obs_times=..., obs_values=...)`),
      simulators pass these values as `obs=...` to the observation `numpyro.sample`
      sites.
    - **Prefer filtering for inference**: for parameter inference that marginalizes
      latent trajectories, prefer filtering (`dynestyx.inference.filters.Filter`)
      over simulators. In particular, conditioning directly on observations with
      `SDESimulator` is usually a poor inference strategy.

!!! note "Deterministic sites"
    When simulator trajectories are produced, sites are recorded as `"{name}_{key}"`
    where `name` is the first
    argument to `dsx.sample(name, dynamics, ...)` (conventionally `"f"`):

    - `"f_times"`: trajectory time grid, shape `(n_sim, T)`,
    - `"f_states"`: latent trajectory, shape `(n_sim, T, state_dim)`,
    - `"f_observations"`: sampled or conditioned emissions, shape `(n_sim, T, obs_dim)`.

    In filter-rollout mode (`predict_times` with filtered posteriors), additional
    keys `"f_predicted_states"`, `"f_predicted_times"`, and
    `"f_predicted_observations"` are recorded.

    Under `numpyro.infer.Predictive(model, num_samples=N)`, NumPyro prepends a leading
    `num_samples` axis, giving final shapes `(num_samples, n_sim, T, dim)`.
    Use `dynestyx.flatten_draws` to collapse the `(num_samples, n_sim)` prefix into one
    axis for plotting or downstream analysis.

    If both `obs_times` and `predict_times` are omitted, no simulation is performed
    and these sites are not added.

## Simulators

::: dynestyx.simulators
    options:
      filters: []
      show_root_heading: false
      show_root_toc_entry: true
  
