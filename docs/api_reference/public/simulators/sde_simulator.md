# SDESimulator

::: dynestyx.simulation.sde.SDESimulator
    options:
      show_root_heading: false
      show_root_toc_entry: true

## Examples

??? example "Predictive with SDESimulator"
    ```python
    import dynestyx as dsx
    import jax.numpy as jnp
    import jax.random as jr
    import numpyro
    import numpyro.distributions as dist
    from dynestyx import (
        ContinuousTimeStateEvolution,
        DynamicalModel,
        FullDiffusion,
        SDESimulator,
    )
    from numpyro.infer import Predictive

    state_dim = 1
    observation_dim = 1
    bm_dim = 1

    def model(predict_times=None):
        theta = numpyro.sample("theta", dist.LogNormal(-0.5, 0.2))
        sigma_x = numpyro.sample("sigma_x", dist.LogNormal(-1.0, 0.2))
        sigma_y = numpyro.sample("sigma_y", dist.LogNormal(-1.5, 0.2))
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(state_dim),
                covariance_matrix=jnp.eye(state_dim),
            ),
            state_evolution=ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -theta * x,
                diffusion=FullDiffusion(sigma_x * jnp.eye(state_dim, bm_dim)),
            ),
            observation_model=lambda x, u, t: dist.MultivariateNormal(
                x,
                sigma_y**2 * jnp.eye(observation_dim),
            ),
        )
        return dsx.sample("f", dynamics, predict_times=predict_times)

    predict_times = jnp.linspace(0.0, 5.0, 51)
    with SDESimulator():
        prior_pred = Predictive(model, num_samples=5)(jr.PRNGKey(0), predict_times=predict_times)
    print("Predictive keys:", sorted(prior_pred.keys()))  # e.g. ['f_observations', 'f_states', 'f_times', 'sigma_x', 'sigma_y', 'theta', ...]
    print("Predictive shapes:", {k: v.shape for k, v in prior_pred.items()})  # trajectory arrays: (num_samples, n_sim, T, dim); here num_samples=5, n_sim=1
    ```

!!! note
    `SDESimulator` is generation-only. Native SDE explicit latent-path
    inference should currently go through a discretization step first, then
    `LatentPathBuilder`; otherwise prefer `Filter` for marginalized inference.
