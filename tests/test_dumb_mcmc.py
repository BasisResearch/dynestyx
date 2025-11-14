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
from dsx.dynamical_models import DynamicalModel
from dsx.ops import sample_ds, Trajectory
from typing import Callable
import jax.random as jr

class DumbParameterizedDynamics(DynamicalModel):
    """Dynamical model that takes parameters passed to a function."""
    
    def __init__(self, func: Callable, params: dict, state_name: str = "x"):
        self.params = params
        self.state_name = state_name
        # Wrap func to apply parameters as kwargs
        self.func = lambda times: func(times, **params)


class SinDynamics(DumbParameterizedDynamics):
    """Sin dynamics with phase parameter."""
    
    def __init__(self, phase: float, state_name: str = "x"):
        def sin_func(times, phase):
            return jnp.sin(times + phase)
        super().__init__(sin_func, {"phase": phase}, state_name)


class DumbSolver(BaseSolver):
    """Solver that works with DumbParameterizedDynamics."""
    
    def solve(self, times, dynamics: DumbParameterizedDynamics) -> States:
        if not isinstance(dynamics, DumbParameterizedDynamics):
            raise NotImplementedError(f"DumbSolver only works with DumbDynamics, got {type(dynamics)}")
        return {dynamics.state_name: dynamics.func(times)}


class MSELogFactorAdder(BaseCDDynamaxLogFactorAdder):
    """Log factor adder that adds MSE between trajectory and observations."""
    
    def add_log_factors(self, name: str, dynamics: DumbParameterizedDynamics, obs: Trajectory):
        # Get trajectory function from solver via fwd()
        # TODO this is a little weird, because what forward actually resolves to is not clear here, (its sample_ds).
        trajectory_fn = fwd()
        
        # Extract observed times and states
        obs_times, obs_states = obs
        
        # Evaluate trajectory at observed times
        pred_states = trajectory_fn(obs_times)
        
        # Compute MSE for each state variable
        mse_total = 0.0
        for state_name in obs_states:
            mse = jnp.mean((pred_states[state_name] - obs_states[state_name]) ** 2)
            mse_total += mse
        
        # Draw randomness from the NumPyro RNG stream managed by NUTS
        key = numpyro.prng_key()            # <- this comes from the seed / MCMC
        log_factor_noise = jr.normal(key)   # will feed this key into random algorithms (e.g. EnKF).

        noisy_mse = mse_total*(1 + 0.1 * log_factor_noise)
        # Example: fold it into the log factor (toy pseudo-PF)
        numpyro.factor(f"{name}_noisy_mse_log_factor", -noisy_mse)


def model():
    """Model that samples phase and uses it in dynamics."""
    phase = numpyro.sample("phase", dist.Normal(0.0, 1.0))
    dynamics = SinDynamics(phase=phase)
    return sample_ds("f", dynamics, None)


def conditioned_model(obs_data: dict[str, Trajectory]):
    """Create a conditioned model with solver, log factor adder, and condition handlers."""

    # TODO instantiating these out here and reusing in run_model broke. Jack why
    # solver_handler = handler(DumbSolver())
    # log_factor_adder_handler = handler(MSELogFactorAdder())
    # condition_handler = handler(Condition(obs_data))
    
    def run_model():

        with handler(DumbSolver()):
            with handler(MSELogFactorAdder()):
                with handler(Condition(obs_data)):
                    return model()
    
    return run_model


def test_forward_sampling_smoke():
    """Smoke test of forward sampling."""
    solver = handler(DumbSolver())
    seeded_model = seed(model, rng_seed=0)
    
    # Trace the model to verify phase is sampled correctly
    with trace() as trace_dict:
        with solver:
            trajectory_fn = seeded_model()
    
    # Check that phase was sampled
    assert "phase" in trace_dict, "Phase should be sampled"
    sampled_phase = trace_dict["phase"]["value"]
    
    # Evaluate trajectory at some times
    times = jnp.array([0.0, 1.0, 2.0])
    result = trajectory_fn(times)
    
    # Verify the trajectory uses the sampled phase
    expected = jnp.sin(times + sampled_phase)
    assert jnp.allclose(result["x"], expected), \
        f"Expected sin(t + {sampled_phase}), got {result['x']}"
    assert result["x"].shape == times.shape

def run_mcmc_inference(true_phase: float = 0.5, num_samples: int = 1000, num_warmup: int = 500):
    """Run MCMC inference on synthetic data."""
    rng_key = random.PRNGKey(0)

    def evaluated_model(times):
        f = model()
        return numpyro.deterministic("feval", f(times))

    # Generate synthetic data using Predictive with ground truth phase
    # Predictive handles seeding internally
    solver = handler(DumbSolver())
    
    true_params = {"phase": jnp.array(true_phase)}
    predictive = Predictive(evaluated_model, params=true_params, num_samples=1)

    # Generate observations at some times
    obs_times = jnp.linspace(-2.0, 0.0, 10)
    
    with solver:
        samples = predictive(rng_key, obs_times)

    obs_states = {"x": samples["feval"]["x"]}
    obs_data = {"f": (obs_times, obs_states)}
    
    # Create conditioned model
    cond_model = conditioned_model(obs_data)
    
    # Run NUTS MCMC
    nuts_kernel = NUTS(cond_model)
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, num_warmup=num_warmup)
    mcmc.run(rng_key)
    
    # Get posterior samples
    posterior_samples = mcmc.get_samples()
    
    # Posterior predictive at evaluation times
    eval_times = jnp.linspace(-2.0, 2.0, 100)
    predictive = Predictive(evaluated_model, posterior_samples)
    with handler(DumbSolver()):
        pred_samples = predictive(rng_key, eval_times)
    
    trajectory_evals = pred_samples["feval"]
    
    return {
        "true_phase": true_phase,
        "posterior_phase": posterior_samples["phase"],
        "posterior_predictive": trajectory_evals["x"],
        "eval_times": eval_times,
        "obs_times": obs_times,
        "obs_data": obs_data,
    }


def test_mcmc_smoke():
    """Smoke test of MCMC inference."""
    result = run_mcmc_inference(true_phase=0.5, num_samples=100, num_warmup=50)
    assert "posterior_phase" in result
    assert "posterior_predictive" in result
    assert len(result["posterior_phase"]) > 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    result = run_mcmc_inference(true_phase=0.5)
    
    # Plot posterior samples of phase
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(result["posterior_phase"], bins=50, alpha=0.7, label="Posterior")
    plt.axvline(result["true_phase"], color="r", linestyle="--", linewidth=2, label="True phase")
    plt.xlabel("Phase")
    plt.ylabel("Frequency")
    plt.title("Posterior Distribution of Phase")
    plt.legend()
    
    # Plot confidence intervals for posterior predictive
    plt.subplot(1, 2, 2)
    pred_samples = result["posterior_predictive"]
    eval_times = result["eval_times"]
    
    # Compute percentiles
    lower = jnp.percentile(pred_samples, 2.5, axis=0)
    upper = jnp.percentile(pred_samples, 97.5, axis=0)
    median = jnp.median(pred_samples, axis=0)
    
    plt.fill_between(eval_times, lower, upper, alpha=0.3, label="95% CI")
    plt.plot(eval_times, median, "b-", label="Median")
    
    # Plot true trajectory
    true_traj = SinDynamics(phase=result["true_phase"]).func(eval_times)
    plt.plot(eval_times, true_traj, "r--", label="True trajectory")
    
    # Plot observations
    obs_times, obs_states = result["obs_data"]["f"]
    plt.scatter(obs_times, obs_states["x"], color="k", s=30, zorder=5, label="Observations")
    
    plt.xlabel("Time")
    plt.ylabel("x")
    plt.title("Posterior Predictive Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.show()

