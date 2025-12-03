import pytest
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive
from numpyro.handlers import seed, trace
import jax.numpy as jnp
from jax import random
from contextlib import nullcontext
from effectful.ops.semantics import handler, fwd, coproduct
from dsx.handlers import BaseSolver, States, BaseCDDynamaxLogFactorAdder, Condition
from dsx.ops import sample_ds, Trajectory
from typing import Callable
import jax.random as jr

from dsx.solvers import SDESolver
from dsx.filters import FilterBasedMarginalLogLikelihood
from dsx.models.lorenz63 import make_L63_SDE_model


def model():
    """Model that samples drift parameter rho and uses it in dynamics."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))
    dynamics = make_L63_SDE_model(rho=rho)
    return sample_ds("f", dynamics, None)




def run_mcmc_inference(true_rho: float = 28.0, num_samples: int = 1000, num_warmup: int = 500):
    """Run MCMC inference on synthetic data."""
    rng_key = random.PRNGKey(0)
    
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = random.split(rng_key, 4)

    
    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(model, params=true_params, num_samples=1, exclude_deterministic=False)

    with handler(SDESolver(key=data_solver_key)):
        with handler(Condition({"times": obs_times})):
            synthetic = predictive(data_init_key)

    obs_states = {"x": synthetic["observations"]}

    # ---------------------------------------------------------
    # Build conditioned model and run NUTS
    # ---------------------------------------------------------    
    def conditioned_model():
            with handler(FilterBasedMarginalLogLikelihood()):
                with handler(Condition({"times": obs_times, "observations": obs_states})):
                    return model()
    
    # Run NUTS MCMC
    nuts_kernel = NUTS(conditioned_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(mcmc_key)
    
    # Get posterior samples
    posterior_samples = mcmc.get_samples()

    # ---------------------------------------------------------
    # Posterior predictive
    # ---------------------------------------------------------
    predictive_post = Predictive(model, posterior_samples)

    with handler(SDESolver()):
        pred = predictive_post(posterior_pred_key)

    return {
        "true_rho": true_rho,
        "posterior_rho": posterior_samples["rho"],
        "posterior_predictive": pred["f"]["x"],
        "obs_times": obs_times,
        "obs_states": obs_states,
    }


# -------------------------------------------------------------
# SMOKE TEST
# -------------------------------------------------------------
def test_mcmc_smoke():
    result = run_mcmc_inference(true_rho=28.0, num_samples=50, num_warmup=50)
    assert "posterior_rho" in result
    assert len(result["posterior_rho"]) > 0
    print("Smoke test passed.")

    
if __name__ == "__main__":
    result = run_mcmc_inference(true_rho=28.0)

    import matplotlib.pyplot as plt

    post = result["posterior_rho"]
    print("Posterior mean rho:", post.mean())

    plt.hist(post, bins=40, alpha=0.7)
    plt.axvline(result["true_rho"], color="r", linestyle="--")
    plt.title("Posterior on rho")
    plt.show()
