# GaussianObservation

::: dynestyx.models.observations.GaussianObservation
    options:
      show_root_heading: false
      show_root_toc_entry: false

!!! note "Structured inference"
    You can always instantiate a model without this structured class (for example,
    by passing a generic callable observation model). But `GaussianObservation`
    explicitly signals Gaussian emission structure that enables fast ensemble
    Kalman methods; see [Filters](../../../inference/filters.md) and `EnKFConfig`
    / `ContinuousTimeEnKFConfig` in
    [FilterConfigs](../../../inference/filter_configs.md).

    Without these exploitable structures, marginalizing latent processes during
    parameter inference typically requires particle filters (`PFConfig` and
    related particle methods), which are often slower.

## Example

??? example "Nonlinear Gaussian observation"
    ```python
    import jax.numpy as jnp
    from dynestyx import GaussianObservation

    def h(x, u, t):
        # Example nonlinear measurement function
        return jnp.array([x[0] ** 2 + 0.1 * x[1]])

    observation = GaussianObservation(
        h=h,
        R=0.05 * jnp.eye(1),
    )

    x_t = jnp.array([1.5, -0.3])
    dist_y = observation(x_t, u=None, t=0.0)  # p(y_t | x_t, t)
    ```
