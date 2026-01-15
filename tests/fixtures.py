import jax.numpy as jnp
import jax.random as jr

from numpyro.infer import Predictive

from effectful.ops.semantics import handler
from dsx.handlers import Condition
from dsx.ops import Trajectory, Context
from dsx.solvers import SDESolver
from dsx.unrollers import DiscreteTimeUnroller, ODEUnroller, ProbabilisticODEUnroller
from dsx.filters import (
    FilterBasedMarginalLogLikelihood,
    FilterBasedHMMMarginalLogLikelihood,
)

from tests.models import (
    discrete_time_l63_model,
    hmm_model,
    continuous_time_stochastic_l63_model,
    continuous_time_deterministic_l63_model,
)
import pytest


@pytest.fixture
def data_conditioned_hmm():
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = jr.split(rng_key, 4)

    # Set true parameters for synthetic data generation
    true_A = jnp.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    true_mu = jnp.array([-10.0, 0.0, 10.0])
    true_sigma = 0.5

    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.1)

    # Generate synthetic data
    true_params = {"A": true_A, "mu": true_mu, "sigma": true_sigma}
    predictive = Predictive(
        hmm_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(observations=Trajectory(times=obs_times))

    # with handler(BaseSolver()): # SHOULD raise error but does not. WHY JACK?
    with handler(DiscreteTimeUnroller()):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    # Prefer indexing rather than squeeze, to keep (T, obs_dim)
    obs_values = synthetic["observations"][0]  # (T,)

    # ---------------------------------------------------------
    # Build conditioned model
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(observations=observation_trajectory)
        with handler(FilterBasedHMMMarginalLogLikelihood()):
            with handler(Condition(context)):
                return hmm_model()

    return data_conditioned_model, true_params, synthetic


@pytest.fixture
def data_conditioned_discrete_time_l63():
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = jr.split(rng_key, 4)

    # Set true parameters for synthetic data generation
    true_rho = 28.0

    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        discrete_time_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(observations=Trajectory(times=obs_times))

    # with handler(BaseSolver()): # SHOULD raise error but does not. WHY JACK?
    with handler(DiscreteTimeUnroller()):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(observations=observation_trajectory)
        with handler(DiscreteTimeUnroller()):
            with handler(Condition(context)):
                return discrete_time_l63_model()

    return data_conditioned_model, true_params, synthetic


@pytest.fixture
def data_conditioned_continuous_time_stochastic_l63():
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = jr.split(rng_key, 4)

    true_rho = 28.0
    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(observations=Trajectory(times=obs_times))
    with handler(SDESolver(key=data_solver_key)):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(observations=observation_trajectory)
        with handler(FilterBasedMarginalLogLikelihood()):
            with handler(Condition(context)):
                return continuous_time_stochastic_l63_model()

    return data_conditioned_model, true_params, synthetic


@pytest.fixture
def data_conditioned_continuous_time_deterministic_l63():
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = jr.split(rng_key, 4)

    true_rho = 28.0
    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=2.0, step=0.001)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_deterministic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(observations=Trajectory(times=obs_times))
    with handler(ODEUnroller()):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(observations=observation_trajectory)
        with handler(ODEUnroller()):
            with handler(Condition(context)):
                return continuous_time_deterministic_l63_model()

    return data_conditioned_model, true_params, synthetic


@pytest.fixture
def data_conditioned_continuous_time_deterministic_l63_with_probabilistic_solver():
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key = jr.split(rng_key, 4)

    true_rho = 28.0
    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=2.0, step=0.01)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_deterministic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(observations=Trajectory(times=obs_times))
    with handler(ProbabilisticODEUnroller()):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(observations=observation_trajectory)
        with handler(ProbabilisticODEUnroller()):
            with handler(Condition(context)):
                return continuous_time_deterministic_l63_model()

    return data_conditioned_model, true_params, synthetic
