import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import Predictive

from dynestyx.dynamical_models import Context, Trajectory
from dynestyx.filters import Filter
from dynestyx.handlers import Condition, Discretizer
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeUKFConfig,
    EKFConfig,
    KFConfig,
    PFConfig,
    UKFConfig,
)
from dynestyx.inference.hmm_filters import HMMConfig
from dynestyx.simulators import (
    DiscreteTimeSimulator,
    Simulator,
)
from tests.models import (
    continuous_time_deterministic_l63_model,
    continuous_time_LTI_gaussian,
    continuous_time_stochastic_l63_model,
    continuous_time_stochastic_l63_model_dirac_obs,
    discrete_time_l63_model,
    discrete_time_lti_model,
    hmm_model,
    jumpy_controls_model,
    jumpy_controls_model_sde,
    stochastic_volatility,
)


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
    with DiscreteTimeSimulator():
        with Condition(context):
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
        with Filter(filter_config=HMMConfig()):
            with Condition(context):
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
    with DiscreteTimeSimulator():
        with Condition(context):
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
        with DiscreteTimeSimulator():
            with Condition(context):
                return discrete_time_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_discrete_time_l63_filter(request):
    """Discrete-time L63 model using Filter with EnKF."""
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    # Set true parameters for synthetic data generation
    true_rho = 28.0

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

    with DiscreteTimeSimulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # Build conditioned model
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with Filter():
            with Condition(context):
                return discrete_time_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_discrete_time_l63_filter_pf(request):
    """Discrete-time L63 model using Filter with bootstrap particle filter."""
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    # Set true parameters for synthetic data generation
    true_rho = 28.0

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

    with DiscreteTimeSimulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # Build conditioned model
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with Filter(filter_config=PFConfig(n_particles=3_000)):
            with Condition(context):
                return discrete_time_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(
    params=[
        (False, "default"),
        (False, "EnKF"),
        # (False, "EKF"), EKF too bad :(
        # (False, "UKF"), UKF too bad :(
        (True, "default"),
        (True, "EnKF"),
        # (True, "EKF"), EKF too bad :(
        # (True, "UKF"), UKF too bad :(
    ]
)
def data_conditioned_continuous_time_stochastic_l63(request):
    use_controls, filter_type = request.param
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
    with Simulator():
        with Condition(context):
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
        config = {
            "default": ContinuousTimeEnKFConfig(),
            "EnKF": ContinuousTimeEnKFConfig(),
        }[filter_type]
        with Filter(filter_config=config):
            with Condition(context):
                return continuous_time_stochastic_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls, filter_type


@pytest.fixture(params=[False, True])
def data_conditioned_continuous_time_l63_dpf(request):
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
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.1)

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
    with Simulator():
        with Condition(context):
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
        with Filter(filter_config=ContinuousTimeDPFConfig(n_particles=1_000)):
            with Condition(context):
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
    with Simulator():
        with Condition(context):
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
        with Simulator():
            with Condition(context):
                return continuous_time_deterministic_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_stochastic_volatility(request):
    """Stochastic volatility with DiscreteTimeSimulator; no controls.
    params: identity_observation (False = noisily observed, True = DiracIdentityObservation)."""
    identity_observation = request.param
    rng_key = jr.PRNGKey(0)
    data_init_key, _mcmc_key, _posterior_pred_key, _ctrl_key = jr.split(rng_key, 4)

    true_phi = 0.9
    obs_times = jnp.arange(start=0.0, stop=100.0, step=1.0)
    control_trajectory = Trajectory()

    def model():
        return stochastic_volatility(identity_observation=identity_observation)

    true_params = {"phi": jnp.array(true_phi)}
    predictive = Predictive(
        model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    with DiscreteTimeSimulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with DiscreteTimeSimulator():
            with Condition(context):
                return stochastic_volatility(identity_observation=identity_observation)

    return data_conditioned_model, true_params, synthetic, identity_observation


@pytest.fixture(
    params=[
        (False, "default"),
        (False, "EnKF"),
        (False, "EKF"),
        (False, "UKF"),
        (True, "default"),
        (True, "EnKF"),
        (True, "EKF"),
        (True, "UKF"),
    ]
)
def data_conditioned_continuous_time_lti_gaussian(request):
    use_controls, filter_type = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    true_rho = 2.0
    obs_times = jnp.arange(start=0.0, stop=10.0, step=0.05)

    # Always generate control trajectory to keep randomness consistent
    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_LTI_gaussian,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    with Simulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        config = {
            "default": ContinuousTimeEnKFConfig(),
            "EnKF": ContinuousTimeEnKFConfig(),
            "EKF": ContinuousTimeEKFConfig(),
            "UKF": ContinuousTimeUKFConfig(),
        }[filter_type]
        with Filter(filter_config=config):
            with Condition(context):
                return continuous_time_LTI_gaussian()

    return data_conditioned_model, true_params, synthetic, use_controls, filter_type


@pytest.fixture(params=[False, True])
def data_conditioned_continuous_time_lti_gaussian_dpf(request):
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    true_rho = 2.0
    obs_times = jnp.arange(start=0.0, stop=10.0, step=0.05)

    # Always generate control trajectory to keep randomness consistent
    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_LTI_gaussian,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    with Simulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with Filter(filter_config=ContinuousTimeDPFConfig(n_particles=2_500)):
            with Condition(context):
                return continuous_time_LTI_gaussian()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_discrete_time_l63_auto_dirac_obs(request):
    """Like data_conditioned_discrete_time_l63_auto but with Dirac full-state obs (L63)."""
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    true_rho = 28.0
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_stochastic_l63_model_dirac_obs,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    with DiscreteTimeSimulator():
        with Discretizer():
            with Condition(context):
                synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)

    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with DiscreteTimeSimulator():
            with Discretizer():
                with Condition(context):
                    return continuous_time_stochastic_l63_model_dirac_obs()

    return data_conditioned_model, true_params, synthetic, use_controls


@pytest.fixture(params=[False, True])
def data_conditioned_discrete_time_l63_auto(request):
    """Uses continuous_time_stochastic_l63_model with Discretize(EulMar) to get
    a discrete-time transition (Euler-Maruyama over each [t_now, t_next]), then
    DiscreteTimeSimulator + Condition."""
    use_controls = request.param
    rng_key = jr.PRNGKey(0)

    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    true_rho = 28.0
    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)

    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    true_params = {"rho": jnp.array(true_rho)}
    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )
    # Order: Condition innermost (injects context), then Discretizer (CTE->discrete),
    # then DiscreteTimeSimulator (simulates with discrete dynamics + context).
    with Simulator():
        with Discretizer():
            with Condition(context):
                synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)

    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        with Simulator():
            with Discretizer():
                with Condition(context):
                    return continuous_time_stochastic_l63_model()

    return data_conditioned_model, true_params, synthetic, use_controls


def data_conditioned_jumpy_controls():
    rng_key = jr.PRNGKey(0)
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )
    predictive = Predictive(
        jumpy_controls_model,
        num_samples=1,
        exclude_deterministic=False,
    )

    obs_times = jnp.arange(start=0.0, stop=20.0, step=0.01)
    controls = jnp.ones((len(obs_times),)) * 100
    for i in range(1, len(controls), 2):
        controls = controls.at[i].set(-controls[i])

    context = Context(
        observations=Trajectory(times=obs_times),
        controls=Trajectory(times=obs_times, values=controls),
    )
    with DiscreteTimeSimulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(
            observations=observation_trajectory,
            controls=Trajectory(times=obs_times, values=controls),
        )
        with Filter(filter_config=EKFConfig(record_filtered_states_mean=True)):
            with Condition(context):
                return jumpy_controls_model()

    return data_conditioned_model, synthetic


def data_conditioned_jumpy_controls_sde():
    rng_key = jr.PRNGKey(0)
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )
    predictive = Predictive(
        jumpy_controls_model_sde,
        num_samples=1,
        exclude_deterministic=False,
    )

    obs_times = jnp.arange(start=0.0, stop=1.0, step=0.01)
    controls = jnp.ones((len(obs_times),)) * 100
    for i in range(1, len(controls), 2):
        controls = controls.at[i].set(-controls[i])

    context = Context(
        observations=Trajectory(times=obs_times),
        controls=Trajectory(times=obs_times, values=controls),
    )
    with Simulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        context = Context(
            observations=observation_trajectory,
            controls=Trajectory(times=obs_times, values=controls),
        )
        with FilterBasedMarginalLogLikelihood(
            filter_config=ContinuousTimeEKFConfig(record_filtered_states_mean=True)
        ):
            with Condition(context):
                return jumpy_controls_model_sde()

    return data_conditioned_model, synthetic


@pytest.fixture(
    params=[
        (uc, ft) for uc in [False, True] for ft in ["kf", "taylor_kf", "ekf", "ukf"]
    ],
    ids=lambda p: f"controls={p[0]},filter={p[1]}",
)
def data_conditioned_discrete_time_lti_kf(request):
    """Discrete-time LTI model using Filter (kf, taylor_kf, ekf, ukf)."""
    use_controls, filter_type = request.param
    rng_key = jr.PRNGKey(0)

    # Always split into 5 keys to keep randomness consistent
    data_init_key, data_solver_key, mcmc_key, posterior_pred_key, ctrl_key = jr.split(
        rng_key, 5
    )

    # Set true parameters for synthetic data generation
    true_alpha = 0.4

    # Generate observations at some times
    obs_times = jnp.arange(start=0.0, stop=20.0, step=1.0)

    # Always generate control trajectory to keep randomness consistent
    control_trajectory = Trajectory()
    if use_controls:
        control_dim = 1
        ctrl_values = jr.normal(ctrl_key, shape=(len(obs_times), control_dim))
        control_trajectory = Trajectory(times=obs_times, values=ctrl_values)

    # Generate synthetic data
    true_params = {"alpha": jnp.array(true_alpha)}
    predictive = Predictive(
        discrete_time_lti_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )

    # Always pass control_trajectory to context (empty Trajectory() if not using controls)
    context = Context(
        observations=Trajectory(times=obs_times), controls=control_trajectory
    )

    with DiscreteTimeSimulator():
        with Condition(context):
            synthetic = predictive(data_init_key)

    obs_values = synthetic["observations"].squeeze(0)  # shape (T, obs_dim)

    # Build conditioned model
    observation_trajectory = Trajectory(times=obs_times, values=obs_values)

    def data_conditioned_model():
        # Always pass control_trajectory to context (empty Trajectory() if not using controls)
        context = Context(
            observations=observation_trajectory, controls=control_trajectory
        )
        config = {
            "kf": KFConfig(),
            "taylor_kf": EKFConfig(filter_source="cuthbert"),
            "ekf": EKFConfig(filter_source="cd_dynamax"),
            "ukf": UKFConfig(),
        }[filter_type]
        with Filter(filter_config=config):
            with Condition(context):
                return discrete_time_lti_model()

    return data_conditioned_model, true_params, synthetic, use_controls, filter_type
