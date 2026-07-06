# LatentPathBuilder

Developer-facing reference for latent-path construction and scoring.

The latent-state refactor separates:

- `dynestyx.inference.latent.builder`: handler logic and NumPyro-site
  registration
- `dynestyx.inference.latent.state_path`: pure-JAX path assembly helpers
- `dynestyx.inference.latent.trajectory_log_probs`: pure-JAX joint trajectory
  scoring helpers

::: dynestyx.inference.latent.builder
    options:
      filters: []
