# LatentPathBuilder

Developer-facing reference for latent-path construction and scoring.

The latent-state refactor separates:

- `dynestyx.inference.latent.base`: handler logic and NumPyro-site
  registration
- `dynestyx.inference.latent.metadata`: latent layout / indexing / canonicalization
- `dynestyx.inference.latent.assembly`: pure-JAX state-path reconstruction
- `dynestyx.inference.latent.scoring`: pure-JAX joint trajectory scoring

::: dynestyx.inference.latent.base
    options:
      filters: []
