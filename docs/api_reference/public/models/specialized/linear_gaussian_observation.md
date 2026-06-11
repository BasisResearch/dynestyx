# LinearGaussianObservation

::: dynestyx.models.observations.LinearGaussianObservation
    options:
      show_root_heading: false
      show_root_toc_entry: false

!!! note "Structured inference"
    You can instantiate equivalent observation behavior without this class
    (for example, with a custom callable). However, this structured linear-Gaussian
    observation form is what lets filtering backends use fast Kalman-family
    methods; see [Filters](../../../inference/filters.md), especially `KFConfig`
    and `EnKFConfig` (or `ContinuousTimeKFConfig` / `ContinuousTimeEnKFConfig`)
    in [FilterConfigs](../../../inference/filter_configs.md).

    Without this exploitable structure, parameter inference that marginalizes
    latent trajectories often relies on particle filters (`PFConfig` and
    related particle methods), which are typically slower.

## Example

??? example "Linear Gaussian observation with control input"
    ```python
    import jax.numpy as jnp
    from dynestyx import LinearGaussianObservation

    observation = LinearGaussianObservation(
        H=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        R=0.1 * jnp.eye(2),
        D=jnp.array([[1.0], [0.5]]),
        bias=jnp.array([0.0, 0.1]),
    )

    x_t = jnp.array([1.2, -0.3])
    u_t = jnp.array([0.8])
    dist_y = observation(x_t, u_t, t=0.0)  # p(y_t | x_t, u_t, t)
    ```

??? example "Time-varying observation model"
    Each parameter may instead be a callable `(t,) -> value` evaluated at the
    observation time. Time-varying models are supported by the simulators and
    by `KFConfig(filter_source="cuthbert")` /
    `KFSmootherConfig(filter_source="cuthbert")`.

    ```python
    import jax.numpy as jnp
    from dynestyx import LinearGaussianObservation


    def observation_matrix(t):
        return jnp.eye(2) * (1.0 + 0.1 * t)


    observation = LinearGaussianObservation(H=observation_matrix, R=0.1 * jnp.eye(2))
    params = observation.params_at(2.5)  # LinearGaussianObservationParams(H=..., ...)
    ```

## LinearGaussianObservationParams

::: dynestyx.models.observations.LinearGaussianObservationParams
    options:
      show_root_heading: false
      show_root_toc_entry: false

