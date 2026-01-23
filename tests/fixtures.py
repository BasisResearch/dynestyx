import jax.numpy as jnp
import jax.random as jr

from numpyro.infer import Predictive

from effectful.ops.semantics import handler
from dsx.handlers import Condition
from dsx.ops import Trajectory, Context
from dsx.solvers import SDESolver
from dsx.unrollers import DiscreteTimeUnroller, ODEUnroller
from dsx.filters import (
    FilterBasedMarginalLogLikelihood,
    FilterBasedHMMMarginalLogLikelihood,
)

from tests.models import (
    discrete_time_l63_model,
    hmm_model,
    continuous_time_stochastic_l63_model,
    continuous_time_deterministic_l63_model,
    continuous_time_linear_sde_model,
)
import pytest


@pytest.fixture(params=[False, True])
def data_conditioned_hmm(request):
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    # Set true parameters for synthetic data generation
    true_A = jnp.array([[0.7, 0.2, 0.1], [0.3, 0.4, 0.3], [0.2, 0.3, 0.5]])
    true_mu = jnp.array([-10.0, 0.0, 10.0])
    true_sigma = 0.5

    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.1)

    # Initialize control trajectory (empty if not using controls)
    control_trajectory = Trajectory()
    if use_controls:
        # Generate controls and set trajectory
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    # Generate synthetic data
    true_params = {"A": true_A, "mu": true_mu, "sigma": true_sigma}
    predictive = Predictive(
        hmm_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )

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
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with handler(FilterBasedHMMMarginalLogLikelihood()):
            with handler(Condition(context)):
                return hmm_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_discrete_time_l63(request):
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    # Set true parameters for synthetic data generation
    true_rho = 28.0

    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    # Always generate control trajectory to keep randomness consistent
    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        discrete_time_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )

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
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with handler(DiscreteTimeUnroller()):
            with handler(Condition(context)):
                return discrete_time_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_continuous_time_stochastic_l63(request):
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    true_rho = 28.0
    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    # Always generate control trajectory to keep randomness consistent
    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    with handler(SDESolver(key=data_solver_key)):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with handler(FilterBasedMarginalLogLikelihood()):
            with handler(Condition(context)):
                return continuous_time_stochastic_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_continuous_time_deterministic_l63(request):
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    true_rho = 28.0
    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=2.0, step=0.001)

    # Always generate control trajectory to keep randomness consistent
    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    # Generate synthetic data
    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_deterministic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    with handler(ODEUnroller()):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with handler(ODEUnroller()):
            with handler(Condition(context)):
                return continuous_time_deterministic_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_continuous_time_linear_sde(request):
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    # Set true parameters for synthetic data generation
    # Use oscillatory + dissipative structure for true A matrix
    true_alpha = 0.3  # dissipation
    true_omega = 1.5  # oscillation frequency
    true_sigma_observation = 0.2  # observation noise scale
    
    # Construct A from oscillatory/dissipative structure
    # A = [[-alpha, -omega],
    #      [omega, -alpha]]
    true_A = jnp.array([[-true_alpha, -true_omega], [true_omega, -true_alpha]])
    
    # B matrix: 2x1 control input matrix
    true_B = jnp.ones((2, 1))
    
    # L matrix: 2x2 diffusion coefficient
    true_L = 0.3 * jnp.eye(2)

    # ---------------------------------------------------------
    # Generate synthetic observations using Predictive
    # ---------------------------------------------------------
    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.1)

    # Always generate control trajectory to keep randomness consistent
    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim)) * 0.1
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    # Generate synthetic data
    true_params = {
        "A": true_A,
        "B": true_B,
        "L": true_L,
        "sigma_observation": jnp.array(true_sigma_observation),
    }
    predictive = Predictive(
        continuous_time_linear_sde_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    with handler(SDESolver(key=data_solver_key)):
        with handler(Condition(context)):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # ---------------------------------------------------------
    # Build conditioned model with EKF
    # ---------------------------------------------------------
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with handler(FilterBasedMarginalLogLikelihood(filter_type="EKF", diffeqsolve_dt0=0.01)):
            with handler(Condition(context)):
                return continuous_time_linear_sde_model()

    return data_conditioned_model, true_params, synthetic, use_controls
