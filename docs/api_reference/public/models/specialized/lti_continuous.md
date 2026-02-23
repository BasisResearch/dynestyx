# LTI_continuous

::: dynestyx.models.lti_dynamics.LTI_continuous
    options:
      show_root_heading: false
      show_root_toc_entry: false

!!! note "Structured inference"
    This factory is a convenience wrapper: you can construct an equivalent
    Kalman-filter-friendly model by manually wiring `DynamicalModel` with an
    affine drift (`AffineDrift`), a constant diffusion coefficient, and
    `LinearGaussianObservation` (emissions), plus a Gaussian initial condition.
    Using these structured classes makes the linear/Gaussian structure explicit
    so `dynestyx` can dispatch to fast Kalman-style filtering in continuous time;
    see [Filters](../../../inference/filters.md) and `ContinuousTimeKFConfig` in
    [FilterConfigs](../../../inference/filter_configs.md).

    Without this exploitable structure, parameter inference that marginalizes
    latent trajectories generally falls back to particle filters
    (`ContinuousTimeDPFConfig` / particle-style methods), which are typically slower.

!!! note "Identifiability and canonical forms"
    With partial observations, standard LTI parameterizations can be non-identifiable,
    leading to multi-modal or poorly behaved posteriors. Canonical/minimal
    parameterizations are often recommended for Bayesian system identification in
    these settings; see [Canonical Bayesian Linear System Identification](https://arxiv.org/abs/2507.11535).

## Example

??? example "Continuous-time LTI model factory"
    ```python
    import jax.numpy as jnp
    from dynestyx import LTI_continuous

    model = LTI_continuous(
        A=jnp.array([[0.0, 1.0], [-1.0, -0.2]]),
        L=0.1 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=0.05 * jnp.eye(1),
    )
    ```

