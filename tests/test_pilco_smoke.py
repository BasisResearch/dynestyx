import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx.pilco import (
    MGPR,
    PILCO,
    ExponentialReward,
    LinearController,
    MomentMatchingPropagator,
    RBFController,
    collect_random_rollout,
)
from dynestyx.pilco.envs import InvertedPendulumEnv
from dynestyx.simulators import DiscreteTimeSimulator


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


STATE_DIM, CONTROL_DIM, N_TRAIN = 2, 1, 30


@pytest.fixture()
def toy_data():
    key = jr.PRNGKey(0)
    k1, k2 = jr.split(key)
    X = jr.normal(k1, (N_TRAIN, STATE_DIM + CONTROL_DIM))
    Y = jr.normal(k2, (N_TRAIN, STATE_DIM))
    return X, Y


@pytest.fixture()
def pilco(toy_data):
    X, Y = toy_data
    controller = LinearController(STATE_DIM, CONTROL_DIM, key=jr.PRNGKey(1))
    reward = ExponentialReward(STATE_DIM, target=jnp.zeros(STATE_DIM))
    return PILCO(X, Y, controller=controller, reward=reward, horizon=5)


def test_moment_matching_mean_vs_monte_carlo(toy_data):
    X, Y = toy_data
    mgpr = MGPR(X, Y)
    m = jnp.zeros(STATE_DIM + CONTROL_DIM)
    s = 0.1 * jnp.eye(STATE_DIM + CONTROL_DIM)
    M_mm, _, _ = mgpr.predict_uncertain(m, s)

    key = jr.PRNGKey(42)
    samples = jr.multivariate_normal(key, m, s, shape=(10_000,))
    mc_means = jnp.array([mgpr._predict_deterministic(x)[0] for x in samples]).mean(
        axis=0
    )

    assert jnp.allclose(M_mm, mc_means, atol=0.05)


def test_gp_log_marginal_likelihood_finite_and_grads(toy_data):
    X, Y = toy_data
    mgpr = MGPR(X, Y)
    lml = mgpr.log_marginal_likelihood()
    assert jnp.isfinite(lml)

    grad_fn = jax.grad(lambda m: m.log_marginal_likelihood())
    grads = grad_fn(mgpr)
    assert jnp.all(jnp.isfinite(grads.log_lengthscales))
    assert jnp.all(jnp.isfinite(grads.log_signal_variance))
    assert jnp.all(jnp.isfinite(grads.log_noise_variance))


def test_pilco_propagate_no_nan(pilco):
    m, s = pilco.m_init, pilco.s_init
    for _ in range(5):
        m, s = pilco.propagate(m, s)
    assert jnp.all(jnp.isfinite(m))
    assert jnp.all(jnp.isfinite(s))


def test_gp_state_evolution_with_simulator(pilco):
    gp_dynamics = pilco.to_dynamical_model()
    obs_times = jnp.arange(5.0)

    def _model(obs_times):
        return dsx.sample("gp", gp_dynamics, obs_times=obs_times)

    with DiscreteTimeSimulator():
        result = Predictive(_model, num_samples=1, exclude_deterministic=False)(
            jr.PRNGKey(0),
            obs_times=obs_times,
        )

    assert "states" in result
    assert result["states"].shape[1:] == (5, STATE_DIM)


def test_moment_matching_propagator_records_sites(pilco):
    model_fn = pilco.make_numpyro_model()
    obs_times = jnp.arange(5.0)

    with MomentMatchingPropagator(pilco=pilco):
        with DiscreteTimeSimulator():
            result = Predictive(model_fn, num_samples=1, exclude_deterministic=False)(
                jr.PRNGKey(0),
                obs_times=obs_times,
            )

    assert "gp_mm_means" in result
    assert "gp_mm_covs_diag" in result
    assert "gp_mm_reward" in result


def test_collect_random_rollout_shapes():
    env = InvertedPendulumEnv()
    dynamics = env.to_discrete_dynamical_model(x0=jnp.zeros(2))
    obs_times = jnp.arange(10.0) * env.dt
    result = collect_random_rollout(
        dynamics,
        obs_times,
        jr.PRNGKey(0),
        max_action=env.max_torque,
        control_dim=1,
    )
    assert "states" in result
    assert result["states"].shape == (len(obs_times), STATE_DIM)


def test_exponential_reward_in_unit_interval():
    reward = ExponentialReward(STATE_DIM, target=jnp.zeros(STATE_DIM))
    m = jnp.ones(STATE_DIM)
    s = 0.5 * jnp.eye(STATE_DIM)
    mu_r, _ = reward(m, s)
    assert 0.0 <= float(mu_r) <= 1.0


def test_linear_controller_shapes():
    ctrl = LinearController(STATE_DIM, CONTROL_DIM, key=jr.PRNGKey(0))
    m = jnp.zeros(STATE_DIM)
    s = jnp.eye(STATE_DIM)
    m_u, s_u, c_xu = ctrl.compute_action(m, s)
    assert m_u.shape == (CONTROL_DIM,)
    assert s_u.shape == (CONTROL_DIM, CONTROL_DIM)
    assert c_xu.shape == (STATE_DIM, CONTROL_DIM)


def test_rbf_controller_shapes():
    ctrl = RBFController(STATE_DIM, CONTROL_DIM, n_basis=10, key=jr.PRNGKey(0))
    m = jnp.zeros(STATE_DIM)
    s = jnp.eye(STATE_DIM)
    m_u, s_u, c_xu = ctrl.compute_action(m, s)
    assert m_u.shape == (CONTROL_DIM,)
    assert s_u.shape == (CONTROL_DIM, CONTROL_DIM)
    assert c_xu.shape == (STATE_DIM, CONTROL_DIM)
