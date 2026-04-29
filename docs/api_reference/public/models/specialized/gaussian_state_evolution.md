# GaussianStateEvolution

::: dynestyx.models.state_evolution.GaussianStateEvolution
    options:
      show_root_heading: false
      show_root_toc_entry: false

!!! note "Structured inference"
    You can represent the same transition behavior without this class
    (for example, as a generic callable). However, this structured Gaussian
    transition form is what lets filtering backends use Gaussian filtering
    methods for nonlinear models; see [Filters](../../../inference/filters.md),
    especially `EKFConfig`, `UKFConfig`, and `EnKFConfig` in
    [FilterConfigs](../../../inference/filter_configs.md).

    Without this exploitable structure, marginalizing latent trajectories
    during parameter inference typically falls back to particle filters
    (`PFConfig` and related particle methods), which are usually slower.

## Example

??? example "Nonlinear Gaussian transition model"
    ```python
    import jax.numpy as jnp
    from dynestyx import GaussianStateEvolution

    def F(x, u, t_now, t_next):
        dt = t_next - t_now
        return jnp.array([
            x[0] + dt * x[1],
            x[1] + dt * jnp.sin(x[0]),
        ])

    transition = GaussianStateEvolution(
        F=F,
        cov=0.05 * jnp.eye(2),
    )

    x_t = jnp.array([0.5, -0.2])
    u_t = None
    dist_next = transition(x_t, u_t, t_now=0.0, t_next=1.0)  # p(x_{t+1} | x_t, u_t, t)
    ```
