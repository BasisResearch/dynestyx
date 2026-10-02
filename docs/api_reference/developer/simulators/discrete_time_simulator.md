# DiscreteTimeSimulator

::: dynestyx.simulation.discrete.DiscreteTimeSimulator
    options:
      show_root_heading: false
      show_root_toc_entry: false

## Examples

??? example "Predictive with DiscreteTimeSimulator"
    ```python
    import dynestyx as dsx
    import jax.numpy as jnp
    import jax.random as jr
    import numpyro
    import numpyro.distributions as dist
    from dynestyx import DynamicalModel, DiscreteTimeSimulator
    from numpyro.infer import Predictive

    state_dim = 1
    observation_dim = 1

    def model(phi=None, predict_times=None):
        phi = numpyro.sample("phi", dist.Uniform(0.0, 1.0), obs=phi)
        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(state_dim),
                covariance_matrix=jnp.eye(state_dim),
            ),
            state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
                loc=phi * x + 0.1 * jnp.sin(x),
                covariance_matrix=0.2**2 * jnp.eye(state_dim),
            ),
            observation_model=lambda x, u, t: dist.MultivariateNormal(
                x,
                0.3**2 * jnp.eye(observation_dim),
            ),
        )
        return dsx.sample("f", dynamics, predict_times=predict_times)

    predict_times = jnp.arange(20.0)
    with DiscreteTimeSimulator():
        prior_pred = Predictive(model, num_samples=5)(
            jr.PRNGKey(0),
            predict_times=predict_times,
        )
    print("Predictive keys:", sorted(prior_pred.keys()))  # e.g. ['f_observations', 'f_states', 'f_times', 'phi', ...]
    print("Predictive shapes:", {k: v.shape for k, v in prior_pred.items()})  # trajectory arrays: (num_samples, n_sim, T, dim); here num_samples=5, n_sim=1
    ```

!!! note
    `DiscreteTimeSimulator` is generation-only. For explicit latent-state
    inference use `LatentPathBuilder`; for marginalized inference use
    `Filter`. You can then wrap the fitted model in `DiscreteTimeSimulator`
    again to generate rollouts at `predict_times`.
