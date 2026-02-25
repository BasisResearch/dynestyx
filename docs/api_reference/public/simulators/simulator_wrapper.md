# Simulator

::: dynestyx.simulators.Simulator
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Examples

??? example "Predictive with auto-routing"
    ```python
    import dynestyx as dsx
    import jax.numpy as jnp
    import jax.random as jr
    import numpyro
    import numpyro.distributions as dist
    from dynestyx import DynamicalModel, GaussianStateEvolution, Simulator
    from numpyro.infer import Predictive

    def model(phi=None, obs_times=None, obs_values=None):
        phi = numpyro.sample("phi", dist.Uniform(0.0, 1.0), obs=phi)
        dynamics = DynamicalModel(
            state_dim=1,
            observation_dim=1,
            control_dim=0,
            initial_condition=dist.Normal(0.0, 1.0),
            state_evolution=GaussianStateEvolution(
                F=lambda x, u, t_now, t_next: phi * x + 0.1 * jnp.sin(x),
                cov=jnp.array([[0.2**2]]),
            ),
            observation_model=lambda x, u, t: dist.Normal(x, 0.3),
        )
        return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    obs_times = jnp.arange(20.0)
    with Simulator():
        prior_pred = Predictive(model, num_samples=5)(
            jr.PRNGKey(0),
            obs_times=obs_times,
        )
    print("Predictive keys:", sorted(prior_pred.keys()))  # e.g. ['f', 'observations', 'phi', 'states', 'times', ...]
    print("Predictive shapes:", {k: v.shape for k, v in prior_pred.items()})  # e.g. first axis is num_samples=5
    ```

??? example "NUTS inference with auto-routing"
    ```python
    import dynestyx as dsx
    import jax.random as jr
    from dynestyx import Simulator
    from numpyro.infer import MCMC, NUTS, Predictive

    # Assume `model`, `obs_times`, and `obs_values` are defined as above.
    def conditioned_model():
        return model(obs_times=obs_times, obs_values=obs_values)

    with Simulator():
        mcmc = MCMC(NUTS(conditioned_model), num_warmup=100, num_samples=100)
        mcmc.run(jr.PRNGKey(1))
        posterior = mcmc.get_samples()
    print("Posterior sample keys:", sorted(posterior.keys()))  # stochastic sites (e.g. parameters, and possibly latent x_* sites)
    print("Posterior sample shapes:", {k: v.shape for k, v in posterior.items()})  # each shape starts with num_samples (here 100)

    # Deterministic trajectory keys like 'states'/'observations' are in posterior predictive output.
    with Simulator():
        post_pred = Predictive(model, posterior_samples=posterior)(
            jr.PRNGKey(2), obs_times=obs_times
        )
    print("Posterior predictive keys:", sorted(post_pred.keys()))  # includes 'states', 'observations', 'times'
    print("Posterior predictive shapes:", {k: v.shape for k, v in post_pred.items()})
    ```
