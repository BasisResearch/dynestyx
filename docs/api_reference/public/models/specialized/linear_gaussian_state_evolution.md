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

??? example "Time-varying transition model"
    Each parameter may instead be a callable `(t_now, t_next) -> value`,
    e.g. the exact discretization of a continuous-time LTI SDE on an
    irregular time grid. Time-varying models are supported by the simulators
    and by `KFConfig(filter_source="cuthbert")` /
    `KFSmootherConfig(filter_source="cuthbert")`.

    ```python
    import jax.numpy as jnp
    import jax.scipy.linalg
    from dynestyx import LinearGaussianStateEvolution

    A_c = jnp.array([[-0.5, 0.4], [0.0, -0.3]])
    Q0 = 0.05 * jnp.eye(2)


    def transition_matrix(t_now, t_next):
        return jax.scipy.linalg.expm(A_c * (t_next - t_now))


    def transition_cov(t_now, t_next):
        return Q0 * (t_next - t_now)


    transition = LinearGaussianStateEvolution(A=transition_matrix, cov=transition_cov)
    params = transition.params_at(0.0, 0.3)  # LinearGaussianParams(A=..., ...)
    ```

## LinearGaussianParams

::: dynestyx.models.state_evolution.LinearGaussianParams
    options:
      show_root_heading: false
      show_root_toc_entry: false

