# LatentPathBuilder

Developer-facing reference for latent-path construction and scoring.

The latent-state refactor separates:

- `dynestyx.inference.latent.builder`: handler logic
- `dynestyx.inference.latent.parameterization`: latent parameterizations
  `z = state_path_params` together with reconstruction `x = g(z)`
- `dynestyx.inference.latent.log_prob`: pure-JAX joint trajectory scoring
- `dynestyx.inference.latent._numpyro`: deferred NumPyro-site registration

::: dynestyx.inference.latent.builder
    options:
      filters: []
