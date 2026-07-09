# LatentPathBuilder

`LatentPathBuilder` is the path-parameter construction handler for workflows
that need explicit latent-state variables rather than marginal likelihoods from
`Filter`/`Smoother`.

It is a NumPyro-facing handler: use it through `dsx.sample(...)` inside a
NumPyro model. For pure-JAX trajectory scoring of fixed latent values, use
`dsx.log_prob(...)` instead.

Conceptually:

- `Simulator` generates trajectories and observations.
- `LatentPathBuilder` constructs `state_path_params`, reconstructs the full
  `state_path`, and evaluates `log p(x, y | ...)`.
- `Filter` and `Smoother` remain the preferred observation-consuming handlers
  when you want latent trajectories marginalized out.

The main output names are intentionally distinct from simulator rollout names:

- `f_state_path` / `f_state_path_times`: explicit latent-path inference outputs
  from `LatentPathBuilder`
- `f_states` / `f_times`: rollout outputs from `Simulator` and `dsx.simulate(...)`

For user code, the recommended import is:

```python
from dynestyx import (
    LatentPathBuilder,
)
```

Latent-path layout preparation happens automatically inside the builder. In
particular, partially missing `DiracIdentityObservation` models use compressed
state latents without requiring extra constructor metadata:

```python
with LatentPathBuilder():
    ...
```

`prepare_latent_path_layout(...)` remains available as a lower-level helper for
inspection and testing, but `LatentPathBuilder` prepares its layout internally
rather than accepting a `latent_path_layout=` override at the `dsx.sample(...)`
site:

```python
layout = dsx.prepare_latent_path_layout(
    dynamics,
    obs_times=obs_times,
    obs_values=obs_values,
)
```

For unsupported partially missing *continuous* observation families, use
explicit missing-observation augmentation. In NumPyro workflows this creates a
second latent block `f_missing_obs_values`, reconstructs
`f_completed_obs_values`, and scores the complete-data observation density:

```python
with LatentPathBuilder(missing_observation_strategy="auto"):
    ...
```

This is the recommended fallback when direct masked-likelihood marginalization
is unavailable, for example with correlated continuous observation families
such as multivariate Student `t` models.

::: dynestyx.inference.latent.builder
    options:
      members:
        - LatentPathBuilder
