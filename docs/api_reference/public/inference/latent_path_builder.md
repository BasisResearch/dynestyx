# LatentPathBuilder

`LatentPathBuilder` is the path-parameter construction handler for workflows
that need explicit latent-state variables rather than marginal likelihoods from
`Filter`/`Smoother`.

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
    prepare_dirac_state_path_metadata,
    prepare_missing_observation_metadata,
)
```

For partially missing `DiracIdentityObservation` models under traced NumPyro
inference, precompute the compression metadata eagerly and pass it into the
builder:

```python
dirac_metadata = prepare_dirac_state_path_metadata(
    dynamics,
    obs_times=obs_times,
    obs_values=obs_values,
)

with LatentPathBuilder(dirac_state_path_metadata=dirac_metadata):
    ...
```

For unsupported partially missing *continuous* observation families, use
explicit missing-observation augmentation. In NumPyro workflows this creates a
second latent block `f_missing_obs_values`, reconstructs
`f_completed_obs_values`, and scores the complete-data observation density:

```python
missing_obs_metadata = prepare_missing_observation_metadata(
    dynamics,
    obs_times=obs_times,
    obs_values=obs_values,
)

with LatentPathBuilder(
    missing_observation_strategy="auto",
    missing_obs_metadata=missing_obs_metadata,
):
    ...
```

This is the recommended fallback when direct masked-likelihood marginalization
is unavailable, for example with correlated continuous observation families
such as multivariate Student `t` models.

::: dynestyx.inference.latent.builder
    options:
      members:
        - LatentPathBuilder
