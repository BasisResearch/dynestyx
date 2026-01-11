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
from dsx.solvers import SDESolver
from dsx.filters import FilterBasedMarginalLogLikelihood


def model():
    """Model that samples drift parameter rho and uses it in dynamics."""
    rho = numpyro.sample("rho", dist.Uniform(10.0, 40.0))

    # Create the dynamical model with sampled rho
    dynamics = DynamicalModel(
        state_dim=3,
        observation_dim=1,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(3), covariance_matrix=20.0**2 * jnp.eye(3)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.array(
                [
                    10.0 * (x[1] - x[0]),
                    x[0] * (rho - x[2]) - x[1],
                    x[0] * x[1] - (8.0 / 3.0) * x[2],
                ]
            ),
            diffusion_coefficient=lambda x, u, t: jnp.eye(3),
            diffusion_covariance=lambda x, u, t: jnp.eye(3),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.0]]), R=jnp.array([[5.0**2]])
        ),
    )

    # TODO: observation_model should simply be dist.MultivariateNormal(...) here,
    # but for now we wrap it in LinearGaussianObservation for so that we can extract
    # H and R later for CD-Dynamax conversion (structure exploiting algorithms).
    # In the future, we will build internal logic to identify linear-gaussian observation models
    # and extract H, R automatically.

    # TODO: Functions for drift, diffusion_coefficient, diffusion_covariance should not
    # require (x, u, t) arguments if they are not used. We can wrap them internally.
    # e.g. diffusion_coefficient=jnp.eye(3)
    # e.g. drift = lambda x: F(x, rho)

    # Return a sampled dynamical model, named "f".
    return sample_ds("f", dynamics)


def run_mcmc_inference(
    true_rho: float = 28.0, num_samples: int = 200, num_warmup: int = 100
):
    """Run MCMC inference on synthetic data."""
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = jr.split(rng_key, 4)

    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        model, params=true_params, num_samples=1, exclude_deterministic=False
    )

    context = Context(solve=Trajectory(times=obs_times))
    with handler(SDESolver(key=data_solver_key)):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model and run NUTS
    # ---------------------------------------------------------
    def data_conditioned_model():
        context = Context(observations=Trajectory(times=obs_times, values=obs_values))
        with handler(FilterBasedMarginalLogLikelihood()):
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

    return {
        "true_rho": true_rho,
        "posterior_rho": posterior_samples["rho"],
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
