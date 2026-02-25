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
    When a simulator runs (i.e., when `obs_times` is provided), it records:
    - `"times"`: the observation-time grid used for unrolling,
    - `"states"`: the latent trajectory on that grid,
    - `"observations"`: sampled (or conditioned) emissions on that grid.

    If `obs_times` is omitted, no simulation is performed and these deterministic
    sites are not added.

## BaseSimulator

::: dynestyx.simulators.BaseSimulator
    options:
      show_root_heading: false
      show_root_toc_entry: false
