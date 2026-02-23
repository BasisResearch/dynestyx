# DynamicalModel

::: dynestyx.models.core.DynamicalModel
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Examples

??? example "Discrete-time dissipation with Poisson observation" 
    ```python
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from dynestyx import DynamicalModel

    state_dim = 1
    observation_dim = 1
    dynamics = DynamicalModel(
        state_dim=state_dim,
        observation_dim=observation_dim,
        initial_condition=dist.Uniform(-1.0, 1.0),
        state_evolution=lambda: x, u, t: dist.MultivariateNormal(loc= 0.9 * x, covariance_matrix = jnp.eye(1))
        observation_model=lambda x, u, t: dist.Poisson(rate=jnp.exp(x)),
    )
    ```

??? example "SDE model with linear Gaussian observation"
    ```python
    import jax.numpy as jnp
    import numpyro.distributions as dist
    from dynestyx import (
        DynamicalModel,
        ContinuousTimeStateEvolution,
        LinearGaussianObservation,
    )

    state_dim = 3
    observation_dim = 1
    bm_dim = 2

    dynamics = DynamicalModel(
        state_dim=state_dim,
        observation_dim=observation_dim,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(state_dim),
            covariance_matrix=jnp.eye(state_dim),
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -x + u,
            diffusion_coefficient=lambda x, u, t: jnp.eye(state_dim, bm_dim),
            bm_dim=bm_dim,
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(observation_dim, state_dim),
            R=jnp.eye(observation_dim),
        ),
    )
    ```

