
import jax.numpy as jnp
import jax.random as jr

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

from effectful.ops.semantics import handler
from dsx.dynamical_models import DynamicalModel, ContinuousTimeStateEvolution
from dsx.observations import LinearGaussianObservation
from dsx.handlers import Condition
from dsx.ops import sample_ds, Trajectory, Context
from dsx.solvers import DiscreteTimeSolver #, #SDESolver
from dsx.filters import FilterBasedHMMMarginalLogLikelihood #, FilterBasedMarginalLogLikelihood

import numpy as np
import matplotlib.pyplot as plt
from dsx.plotters import plot_hmm_states_and_observations

def model():
    K = 3  # number of discrete states

    # -------------------------------------------------
    # Transition matrix
    # -------------------------------------------------
    A = numpyro.sample(
        "A",
        dist.Dirichlet(jnp.ones(K)).expand([K]).to_event(1)
    )  # shape (K, K)

    # -------------------------------------------------
    # Emission parameters
    # -------------------------------------------------
    mu = numpyro.sample(
        "mu",
        dist.Normal(0.0, 10.0).expand([K]).to_event(1)
    )  # shape (K,)

    sigma = numpyro.sample("sigma", dist.Uniform(0.1, 2.0))

    # -------------------------------------------------
    # Initial condition
    # -------------------------------------------------
    initial_condition = dist.Categorical(
        probs=jnp.ones(K) / K
    )

    # -------------------------------------------------
    # State evolution
    # -------------------------------------------------
    def state_evolution(x, u, t):
        # x is an integer in {0, ..., K-1}
        return dist.Categorical(probs=A[x])

    # -------------------------------------------------
    # Observation model
    # -------------------------------------------------
    def observation_model(x, u, t):
        # y_t | x_t ~ N(mu[x_t], sigma^2)
        return dist.Normal(mu[x], sigma)

    dynamics = DynamicalModel(
        state_dim=K,
        observation_dim=1,
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
    )

    return sample_ds("f", dynamics)

def run_mcmc_inference(num_samples: int = 200, num_warmup: int = 100):
    """Run MCMC inference on synthetic data."""
    rng_key = jr.PRNGKey(0)
    
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = jr.split(rng_key, 4)

    # Set true parameters for synthetic data generation
    true_A = jnp.array([[0.7, 0.2, 0.1],
                        [0.3, 0.4, 0.3],
                        [0.2, 0.3, 0.5]])
    true_mu = jnp.array([-10.0, 0.0, 10.0])
    true_sigma = 0.5
    
    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.1)
 
    # Generate synthetic data
    true_params = {"A": true_A,
                   "mu": true_mu,
                   "sigma": true_sigma}
    predictive = Predictive(model, params=true_params, num_samples=1, exclude_deterministic=False)

    context = Context(solve=Trajectory(times=obs_times))
    
    # with handler(BaseSolver()): # SHOULD raise error but does not. WHY JACK?
    with handler(DiscreteTimeSolver(key=data_solver_key)):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    # Prefer indexing rather than squeeze, to keep (T, obs_dim)
    obs_values = synthetic["observations"][0]  # (T,)
    
    plot_hmm_states_and_observations(
        obs_times,
        synthetic["states"][0],
        obs_values,
        show_fig=True,
    )

    # ---------------------------------------------------------
    # Build conditioned model and run NUTS
    # ---------------------------------------------------------    
    def data_conditioned_model():
        context = Context(observations=Trajectory(times=obs_times, values=obs_values))
        with handler(FilterBasedHMMMarginalLogLikelihood()):
            with handler(Condition(context)):
                return model()
    
    # Run NUTS MCMC
    nuts_kernel = NUTS(data_conditioned_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(mcmc_key)
    
    # Get posterior samples
    posterior_samples = mcmc.get_samples()

    # ---------------------------------------------------------
    # Posterior predictive
    # ---------------------------------------------------------
    # predictive_post = Predictive(model, posterior_samples)

    # with handler(SDESolver()):
    #     pred = predictive_post(posterior_pred_key)

    # return {
    #     "true_rho": true_rho,
    #     "posterior_rho": posterior_samples["rho"],
    # }
    
    return {
        "true_A": true_A,
        "true_mu": true_mu,
        "true_sigma": true_sigma,
        "posterior_A": posterior_samples["A"],
        "posterior_mu": posterior_samples["mu"],
        "posterior_sigma": posterior_samples["sigma"],
    }

# -------------------------------------------------------------    
if __name__ == "__main__":
    result = run_mcmc_inference(num_samples=1000, num_warmup=2000)
    
    # Note: performs well with observation noise sd = 1.0
    # Performs poorly with observation noise sd = 5.0
    # Also, we should be able to speed up this discrete time model.
    # Currently using numpyro's nscan to step through the time sequence.
    # Should probably be doing something else, but lax.scan isn't working!
    
    import arviz as az
    import numpy as np

    def add_chain_dim(x):
        # Converts (draw, ...) -> (chain=1, draw, ...)
        return np.expand_dims(x, axis=0)

    idata = az.from_dict(
        posterior={
            "mu": add_chain_dim(result["posterior_mu"]),        # (1, draw, K)
            "sigma": add_chain_dim(result["posterior_sigma"]),  # (1, draw)
            "A": add_chain_dim(result["posterior_A"]),          # (1, draw, K, K)
        },
        coords={
            "state": [0, 1, 2],
            "state_row": [0, 1, 2],
            "state_col": [0, 1, 2],
            "observation": [],
        },
        dims={
            "mu": ["state"],
            "A": ["state_row", "state_col"],
            "sigma": [],
        },
    )

    az.plot_posterior(
        idata,
        var_names=["sigma"],
        hdi_prob=0.9,
        ref_val=result["true_sigma"],
    )
    plt.show()
    
    az.plot_forest(
        idata,
        var_names=["mu"],
        hdi_prob=0.9,
        # ref_val=result["true_mu"],
    )
    plt.show()

    az.plot_forest(
        idata,
        var_names=["A"],
        hdi_prob=0.9,
        # ref_val=result["true_A"],
    )    
    plt.show()
