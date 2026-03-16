# Overview

Simulators (also called *unrollers*) turn a `DynamicalModel` into explicit NumPyro
`sample` sites for latent states and observations on a provided time grid.

!!! note "Context"
    - **NumPyro context required**: simulators call `numpyro.sample(...)` and draw
      randomness via NumPyro PRNG keys, so they must run inside a NumPyro model
      (or a `numpyro.handlers.seed(...)` context).
    - **`obs_times` is required**: simulators only run when observation times are
      provided (e.g. `dsx.sample(..., DynamicalModel(...), obs_times=...)`), because
      those times define the trajectory grid.
    - **Conditioning is optional**: if `obs_values` is provided (e.g.
      `dsx.sample(..., DynamicalModel(...), obs_times=..., obs_values=...)`),
      simulators pass these values as `obs=...` to the observation `numpyro.sample`
      sites.
    - **Prefer filtering for inference**: for parameter inference that marginalizes
      latent trajectories, prefer filtering (`dynestyx.inference.filters.Filter`)
      over simulators. In particular, conditioning directly on observations with
      `SDESimulator` is usually a poor inference strategy.

!!! note "Deterministic sites"
    When a simulator runs (i.e., when `obs_times` or `predict_times` is provided),
    it records deterministic sites named `"{name}_{key}"` where `name` is the first
    argument to `dsx.sample(name, dynamics, ...)` (conventionally `"f"`):

    - `"f_times"`: the time grid, shape `(n_sim, T)`,
    - `"f_states"`: the latent trajectory, shape `(n_sim, T, state_dim)`,
    - `"f_observations"`: sampled or conditioned emissions, shape `(n_sim, T, obs_dim)`.

    If `predict_times` is provided alongside filtered posteriors (filter-rollout mode),
    the additional keys `"f_predicted_states"`, `"f_predicted_times"`, and
    `"f_predicted_observations"` are also recorded.

    Under `numpyro.infer.Predictive(model, num_samples=N)`, NumPyro prepends a leading
    `num_samples` axis, giving final shapes `(num_samples, n_sim, T, dim)`.
    Use `dynestyx.flatten_draws` to collapse the `(num_samples, n_sim)` prefix into one
    axis for plotting or downstream analysis.

    If `obs_times` is omitted, no simulation is performed and these sites are not added.

## Simulators

::: dynestyx.simulators
    options:
      filters: []
      show_root_heading: false
      show_root_toc_entry: true
  
