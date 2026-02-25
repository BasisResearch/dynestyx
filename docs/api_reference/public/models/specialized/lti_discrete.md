# LTI_discrete

::: dynestyx.models.lti_dynamics.LTI_discrete
    options:
      show_root_heading: false
      show_root_toc_entry: false

!!! note "Structured inference"
    This factory is a convenience wrapper: you can construct an equivalent
    Kalman-filter-friendly model by manually wiring `DynamicalModel` with
    `LinearGaussianStateEvolution` (state transition) and
    `LinearGaussianObservation` (emissions), plus a Gaussian initial condition.
    Using these structured classes makes the linear/Gaussian structure explicit
    so `dynestyx` can dispatch to fast Kalman filtering; see
    [Filters](../../../inference/filters.md) and `KFConfig` in
    [FilterConfigs](../../../inference/filter_configs.md).

    Without this exploitable structure, parameter inference that marginalizes
    latent trajectories generally falls back to particle filters (`PFConfig`),
    which are typically slower.

!!! note "Identifiability and canonical forms"
    With partial observations, standard LTI parameterizations can be non-identifiable,
    leading to multi-modal or poorly behaved posteriors. Canonical/minimal
    parameterizations are often recommended for Bayesian system identification in
    these settings; see [Canonical Bayesian Linear System Identification](https://arxiv.org/abs/2507.11535).

## Example

??? example "Discrete-time LTI model factory"
    ```python
    import jax.numpy as jnp
    from dynestyx import LTI_discrete

    model = LTI_discrete(
        A=jnp.array([[1.0, 0.1], [0.0, 1.0]]),
        Q=0.01 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=0.05 * jnp.eye(1),
    )
    ```

