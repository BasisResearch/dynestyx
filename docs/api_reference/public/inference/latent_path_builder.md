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

Latent coordinates are registered as zero-log-density improper NumPyro sites.
Their forward sampler is nevertheless meaningful: it draws from the actual
dynamical prior instead of from an artificial Gaussian. Consequently:

- discrete `state_path_params` form a coherent initial-condition/transition
  draw;
- ODE `state_path_params` are drawn from the initial-condition prior and then
  solved on the requested time grid;
- exact partially observed paths simulate densely and retain only their free
  coordinates; and
- augmented `missing_obs_values` are drawn from the observation model
  conditional on the assembled state path.

The complete `log p(x, y | ...)` target is registered once as
`f_joint_log_prob_factor`. The improper sites themselves contribute exactly
zero, so there are no Gaussian base-density correction sites.

This also makes bare NumPyro `Predictive` useful: without posterior samples it
returns dynamical-prior paths. With posterior samples it continues to replay the
supplied latent values.

Missing-coordinate preparation happens automatically inside the builder. In
particular:

- partially missing `DiracIdentityObservation` models use per-coordinate
  `state_path_params` without extra constructor metadata;
- unsupported partially missing continuous observation families can fall back
  to explicit observation augmentation through `f_missing_obs_values`; and
- the user-facing entry point remains the same:

```python
with LatentPathBuilder():
    ...
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
