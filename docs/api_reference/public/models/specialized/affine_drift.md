# AffineDrift

::: dynestyx.models.state_evolution.AffineDrift
    options:
      show_root_heading: false
      show_root_toc_entry: false
      filters:
        - "!^__init__$"

!!! note "Structured inference"
    `AffineDrift` is primarily a convenience class for expressing a common drift
    structure. By itself it does **not** currently trigger a structured inference
    backend.

    Structured filtering typically requires a full set of compatible structure
    (e.g., the full `LTI_continuous` setup pairing an affine drift with linear-Gaussian
    emissions and appropriate noise assumptions); see [Filters](../../../inference/filters.md)
    and [FilterConfigs](../../../inference/filter_configs.md).

    In the future, `AffineDrift` may become directly useful for structured inference
    if we add Rao–Blackwellized methods that can exploit partial linear/Gaussian
    structure.

## Example

??? example "Ornstein–Uhlenbeck (OU) process"
    ```python
    import jax.numpy as jnp
    from dynestyx import AffineDrift, ContinuousTimeStateEvolution

    # OU SDE: dX_t = -theta (X_t - mu) dt + sigma dW_t
    theta = 0.7
    mu = 1.5
    sigma = 0.2

    # Write drift as affine map: f(x, u, t) = A x + b with A = -theta, b = theta * mu
    drift = AffineDrift(A=jnp.array([[-theta]]), b=jnp.array([theta * mu]))

    ou_sde = ContinuousTimeStateEvolution(
        drift=drift,
        diffusion_coefficient=lambda x, u, t: jnp.array([[sigma]]),
    )
    ```
