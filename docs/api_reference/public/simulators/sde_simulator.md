# SDESimulator

::: dynestyx.simulators.SDESimulator
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
    from dynestyx import ContinuousTimeStateEvolution, DynamicalModel, SDESimulator
    from numpyro.infer import Predictive

    state_dim = 1
    observation_dim = 1
    bm_dim = 1

    def model(obs_times=None, obs_values=None):
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
                diffusion_coefficient=lambda x, u, t: sigma_x * jnp.eye(state_dim, bm_dim),
            ),
            observation_model=lambda x, u, t: dist.MultivariateNormal(
                x,
                sigma_y**2 * jnp.eye(observation_dim),
            ),
        )
        return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    obs_times = jnp.linspace(0.0, 5.0, 51)
    with SDESimulator():
        prior_pred = Predictive(model, num_samples=5)(jr.PRNGKey(0), predict_times=obs_times)
    print("Predictive keys:", sorted(prior_pred.keys()))  # e.g. ['f_observations', 'f_states', 'f_times', 'sigma_x', 'sigma_y', 'theta', ...]
    print("Predictive shapes:", {k: v.shape for k, v in prior_pred.items()})  # e.g. first axis is num_samples=5
    ```

??? example "NUTS with SDESimulator (small demonstration)"
    ```python
    import jax.random as jr
    from dynestyx import SDESimulator
    from numpyro.infer import MCMC, NUTS, Predictive

    # Assume `model`, `obs_times`, and `obs_values` are defined as above.
    # Note: this can be expensive; filtering is often preferred for inference.
    def conditioned_model():
        return model(obs_times=obs_times, obs_values=obs_values)

    with SDESimulator():
        mcmc = MCMC(NUTS(conditioned_model), num_warmup=50, num_samples=50)
        mcmc.run(jr.PRNGKey(1))
        posterior = mcmc.get_samples()
    print("Posterior sample keys:", sorted(posterior.keys()))  # stochastic sites (typically parameters and x_0)
    print("Posterior sample shapes:", {k: v.shape for k, v in posterior.items()})

    # Deterministic trajectories are exposed as 'states'/'observations' in posterior predictive output.
    with SDESimulator():
        post_pred = Predictive(model, posterior_samples=posterior)(
            jr.PRNGKey(2), predict_times=obs_times
        )
    print("Posterior predictive keys:", sorted(post_pred.keys()))  # includes 'f_states', 'f_observations', 'f_times'
    print("Posterior predictive shapes:", {k: v.shape for k, v in post_pred.items()})
    ```
