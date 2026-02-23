# LinearGaussianStateEvolution

::: dynestyx.models.state_evolution.LinearGaussianStateEvolution
    options:
      show_root_heading: false
      show_root_toc_entry: false

!!! note "Structured inference"
    You can represent the same transition dynamics without this class
    (for example, as a generic callable). However, this structured linear-Gaussian
    transition form is what enables fast Kalman-family filtering methods; see
    [Filters](../../../inference/filters.md), especially `KFConfig` (and
    `ContinuousTimeKFConfig` for continuous-time settings) in
    [FilterConfigs](../../../inference/filter_configs.md).

    Without this exploitable structure, marginalizing latent trajectories
    during parameter inference typically falls back to particle filters
    (`PFConfig` and related particle methods), which are usually slower.

## Example

??? example "Linear Gaussian transition model"
    ```python
    import jax.numpy as jnp
    from dynestyx import LinearGaussianStateEvolution

    transition = LinearGaussianStateEvolution(
        A=jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        cov=0.05 * jnp.eye(2),
        B=jnp.array([[0.0], [1.0]]),
        bias=jnp.array([0.0, 0.0]),
    )

    x_t = jnp.array([0.5, -0.2])
    u_t = jnp.array([0.3])
    dist_next = transition(x_t, u_t, t_now=0.0, t_next=1.0)  # p(x_{t+1} | x_t, u_t, t)
    ```

