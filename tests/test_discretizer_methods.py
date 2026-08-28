"""Minimal tests for the configuration-driven continuous-time discretizers."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.discretizers import (
    DiffraxSampleConfig,
    Discretizer,
    ExactAffineConfig,
    LocalLinearizationConfig,
    MeanTrajectoryLinearizationConfig,
    ODEFlowConfig,
    _discretize_state_evolution,
)
from dynestyx.evaluation.configs import ObservationScoringConfig
from dynestyx.evaluation.handlers import Evaluation
from dynestyx.evaluation.scoring import EnergyScore, GaussianLogProbScore
from dynestyx.inference.configs.filter import EnKFConfig, PFConfig
from dynestyx.inference.configs.simulator import (
    ODESimulatorConfig,
    SDESimulatorConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DeterministicContinuousTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    LinearGaussianObservation,
    StochasticContinuousTimeStateEvolution,
)


def _ode_state_evolution():
    model = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.array([-0.4 * x[0] + u[0], 0.2 * x[1]])
        ),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=jnp.eye(2)),
        control_dim=1,
    )
    assert isinstance(model.state_evolution, DeterministicContinuousTimeStateEvolution)
    return model.state_evolution


def _affine_model() -> DynamicalModel:
    return dsx.LTI_continuous(
        A=jnp.array([[-0.7]]),
        L=jnp.array([[0.4]]),
        H=jnp.ones((1, 1)),
        R=jnp.eye(1),
    )


def _nonlinear_model() -> DynamicalModel:
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.2 * x + 0.1 * x**2,
            diffusion=FullDiffusion(
                lambda x, u, t: jnp.array([[0.3 + 0.05 * jnp.tanh(x[0])]]),
                bm_dim=1,
            ),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.ones((1, 1)),
            R=jnp.eye(1),
        ),
    )


def _ode_config() -> ODESimulatorConfig:
    return ODESimulatorConfig(
        solver=dfx.Tsit5(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.005,
    )


def _diffrax_config() -> DiffraxSampleConfig:
    return DiffraxSampleConfig(
        SDESimulatorConfig(
            source="diffrax",
            solver=dfx.Euler(),
            dt0=0.01,
            max_steps=100,
        )
    )


def test_ode_flow_matches_controlled_linear_analytic_solution():
    evolution = _discretize_state_evolution(
        _ode_state_evolution(),
        ODEFlowConfig(simulator_config=_ode_config()),
    )
    x = jnp.array([1.2, -0.7])
    u = jnp.array([0.3])
    h = 0.35

    transition = evolution(x, u, 0.2, 0.2 + h)

    expected = jnp.array(
        [
            jnp.exp(-0.4 * h) * x[0] + (1.0 - jnp.exp(-0.4 * h)) * u[0] / 0.4,
            jnp.exp(0.2 * h) * x[1],
        ]
    )
    assert isinstance(transition, dist.Delta)
    assert transition.event_shape == (2,)
    assert jnp.allclose(transition.mean, expected, rtol=2e-5, atol=2e-5)


def test_ode_flow_positive_jitter_returns_independent_normal():
    evolution = _discretize_state_evolution(
        _ode_state_evolution(),
        ODEFlowConfig(simulator_config=_ode_config(), jitter_scale=0.3),
    )

    transition = evolution(jnp.array([1.2, -0.7]), jnp.array([0.3]), 0.2, 0.55)

    assert isinstance(transition, dist.Independent)
    assert isinstance(transition.base_dist, dist.Normal)
    assert transition.event_shape == (2,)
    assert jnp.allclose(transition.base_dist.scale, jnp.full((2,), 0.3))
    assert jnp.allclose(transition.variance, jnp.full((2,), 0.09))


def test_ode_flow_passes_simulator_config_settings(monkeypatch):
    simulator_config = ODESimulatorConfig(
        solver=dfx.Euler(),
        adjoint=dfx.DirectAdjoint(),
        stepsize_controller=dfx.ConstantStepSize(),
        dt0=0.02,
        max_steps=37,
        throw=False,
    )
    captured = {}

    def fake_solve(state_evolution, **kwargs):
        del state_evolution
        captured.update(kwargs["diffeqsolve_settings"])
        return kwargs["initial_state"]

    monkeypatch.setattr(
        "dynestyx.discretization.ode_flow.solve_ode_interval", fake_solve
    )
    evolution = _discretize_state_evolution(
        _ode_state_evolution(),
        ODEFlowConfig(simulator_config=simulator_config),
    )

    evolution(jnp.ones(2), jnp.ones(1), 0.0, 0.1)

    assert isinstance(captured["solver"], dfx.Euler)
    assert isinstance(captured["adjoint"], dfx.DirectAdjoint)
    assert isinstance(captured["stepsize_controller"], dfx.ConstantStepSize)
    assert jnp.array_equal(captured["dt0"], jnp.asarray(0.02))
    assert captured["max_steps"] == 37
    assert captured["throw"] is False


@pytest.mark.parametrize("jitter_scale", [-1.0, jnp.inf, jnp.nan])
def test_ode_flow_rejects_invalid_jitter_scale(jitter_scale):
    with pytest.raises(ValueError, match="jitter_scale"):
        ODEFlowConfig(jitter_scale=float(jitter_scale))


def test_ode_flow_automatic_routing_and_config_type_errors():
    ode = _ode_state_evolution()
    assert isinstance(
        _discretize_state_evolution(ode)(jnp.ones(2), jnp.ones(1), 0.0, 0.01),
        dist.Delta,
    )

    with pytest.raises(TypeError, match="requires a stochastic"):
        _discretize_state_evolution(ode, ExactAffineConfig())
    with pytest.raises(TypeError, match="requires a deterministic"):
        _discretize_state_evolution(
            _nonlinear_model().state_evolution,
            ODEFlowConfig(),
        )


def test_ode_flow_is_jittable_and_differentiable():
    evolution = _discretize_state_evolution(
        _ode_state_evolution(),
        ODEFlowConfig(simulator_config=_ode_config()),
    )

    @jax.jit
    def endpoint(initial_value):
        return evolution(
            jnp.array([initial_value, -0.7]), jnp.array([0.3]), 0.2, 0.55
        ).mean[0]

    assert jnp.isfinite(endpoint(1.2))
    assert jnp.allclose(jax.grad(endpoint)(1.2), jnp.exp(-0.4 * 0.35), rtol=2e-5)


def test_ode_flow_discretizer_runs_discrete_simulator():
    def model(predict_times=None):
        dynamics = DynamicalModel(
            initial_condition=dist.MultivariateNormal(jnp.ones(1), 0.1 * jnp.eye(1)),
            state_evolution=ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -0.2 * x
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.ones((1, 1)), R=0.1 * jnp.eye(1)
            ),
        )
        return dsx.sample("f", dynamics, predict_times=predict_times)

    with dsx.DiscreteTimeSimulator():
        with Discretizer(ODEFlowConfig(jitter_scale=0.01)):
            with trace() as tr, seed(rng_seed=2):
                model(jnp.array([0.0, 0.05, 0.12]))

    assert tr["f_states"]["value"].shape == (1, 3, 1)
    assert jnp.all(jnp.isfinite(tr["f_states"]["value"]))


@pytest.mark.parametrize(
    "filter_config",
    [
        EnKFConfig(n_particles=8, crn_seed=jr.PRNGKey(4)),
        PFConfig(n_particles=8, crn_seed=jr.PRNGKey(4)),
    ],
)
def test_ode_flow_runs_sample_based_filters(filter_config):
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.ones(1), 0.1 * jnp.eye(1)),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: -0.2 * x),
        observation_model=LinearGaussianObservation(
            H=jnp.ones((1, 1)), R=0.1 * jnp.eye(1)
        ),
    )

    def model(obs_times, obs_values):
        return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    with Filter(filter_config):
        with Discretizer(ODEFlowConfig(jitter_scale=0.01)):
            with trace() as tr, seed(rng_seed=2):
                model(jnp.array([0.0, 0.05, 0.12]), jnp.ones((3, 1)))

    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])


def test_ode_flow_cuthbert_enkf_supports_observation_scoring():
    obs_times = jnp.array([0.0, 0.05, 0.12])
    obs_values = jnp.array([1.0, 0.9, 0.8])
    n_particles = 12
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.ones(1), 0.1 * jnp.eye(1)),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: -0.2 * x),
        observation_model=LinearGaussianObservation(
            H=jnp.ones((1, 1)),
            R=0.1 * jnp.eye(1),
        ),
    )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(6)):
        with Evaluation(
            observation_scoring_config=ObservationScoringConfig(
                rules=(GaussianLogProbScore(), EnergyScore(beta=1.0)),
                sample_seed=7,
            )
        ):
            with Filter(
                EnKFConfig(
                    n_particles=n_particles,
                    crn_seed=jr.PRNGKey(8),
                )
            ):
                with Discretizer(
                    ODEFlowConfig(
                        simulator_config=_ode_config(),
                        jitter_scale=0.0,
                    )
                ):
                    result = dsx.sample(
                        "f",
                        dynamics,
                        obs_times=obs_times,
                        obs_values=obs_values,
                    )

    predictions = result.predicted_observations
    scores = result.evaluation_result.observation_scores
    assert predictions is not None
    assert predictions.ensemble.shape == (
        len(obs_times),
        n_particles,
        1,
    )
    assert predictions.mean.shape == (len(obs_times), 1)
    assert predictions.cov.shape == (len(obs_times), 1, 1)
    assert scores["gaussian_log_prob"].shape == (len(obs_times), 1)
    assert scores["energy_score"].shape == (len(obs_times), 1)
    assert jnp.all(jnp.isfinite(scores["gaussian_log_prob"]))
    assert jnp.all(jnp.isfinite(scores["energy_score"]))

    cumulative_loglik = result.states.log_normalizing_constant
    per_step_loglik = jnp.diff(
        jnp.concatenate([jnp.zeros_like(cumulative_loglik[:1]), cumulative_loglik])
    )
    assert jnp.allclose(
        scores["gaussian_log_prob"][:, 0],
        per_step_loglik,
        rtol=2e-5,
        atol=2e-5,
    )
    assert jnp.allclose(
        jnp.sum(scores["gaussian_log_prob"]),
        result.marginal_loglik,
        rtol=2e-5,
        atol=2e-5,
    )
    assert "f_predicted_observations_ensemble" in tr


def test_exact_affine_matches_scalar_ou_transition():
    evolution = _discretize_state_evolution(
        _affine_model().state_evolution,
        ExactAffineConfig(),
    )
    x = jnp.array([1.2])
    h = 0.3
    transition = evolution(x, None, 0.0, h)

    expected_decay = jnp.exp(-0.7 * h)
    expected_cov = 0.4**2 * jnp.expm1(-1.4 * h) / -1.4
    assert jnp.allclose(transition.mean, expected_decay * x)
    assert jnp.allclose(transition.covariance_matrix, expected_cov[None, None])


def test_exact_affine_stiff_covariance_is_finite_in_float32():
    model = dsx.LTI_continuous(
        A=jnp.array([[-100.0]], dtype=jnp.float32),
        L=jnp.ones((1, 1), dtype=jnp.float32),
        H=jnp.ones((1, 1), dtype=jnp.float32),
        R=jnp.eye(1, dtype=jnp.float32),
    )
    evolution = _discretize_state_evolution(
        model.state_evolution,
        ExactAffineConfig(),
    )

    transition = evolution(jnp.ones(1, dtype=jnp.float32), None, 0.0, 1.0)

    expected_cov = -jnp.expm1(jnp.asarray(-200.0, dtype=jnp.float32)) / 200.0
    assert jnp.all(jnp.isfinite(transition.covariance_matrix))
    assert jnp.allclose(
        transition.covariance_matrix,
        expected_cov[None, None],
        rtol=2e-5,
    )


def test_exact_affine_integrator_covariance_remains_exact():
    model = dsx.LTI_continuous(
        A=jnp.zeros((1, 1)),
        L=jnp.array([[0.4]]),
        H=jnp.ones((1, 1)),
        R=jnp.eye(1),
    )
    evolution = _discretize_state_evolution(
        model.state_evolution,
        ExactAffineConfig(),
    )
    h = 0.3

    transition = evolution(jnp.array([1.2]), None, 0.0, h)

    assert jnp.allclose(transition.mean, jnp.array([1.2]))
    assert jnp.allclose(transition.covariance_matrix, jnp.array([[h * 0.4**2]]))


def test_local_linearization_stiff_covariance_is_finite_in_float32():
    evolution = _discretize_state_evolution(
        StochasticContinuousTimeStateEvolution(
            drift=lambda x, u, t: -100.0 * x,
            diffusion=FullDiffusion(jnp.ones((1, 1), dtype=jnp.float32)),
        ),
        LocalLinearizationConfig(),
    )

    transition = evolution(jnp.ones(1, dtype=jnp.float32), None, 0.0, 1.0)

    assert jnp.all(jnp.isfinite(transition.covariance_matrix))
    assert jnp.allclose(transition.covariance_matrix, 0.005, rtol=2e-5)


@pytest.mark.parametrize(
    "config",
    [
        LocalLinearizationConfig(),
        MeanTrajectoryLinearizationConfig(ode_solver=_ode_config()),
    ],
)
def test_gaussian_approximations_match_affine_transition(config):
    model = _affine_model()
    exact = _discretize_state_evolution(
        model.state_evolution,
        ExactAffineConfig(),
    )(jnp.array([1.2]), None, 0.0, 0.2)
    approximate = _discretize_state_evolution(
        model.state_evolution,
        config,
    )(jnp.array([1.2]), None, 0.0, 0.2)

    assert jnp.allclose(approximate.mean, exact.mean, rtol=3e-4)
    assert jnp.allclose(
        approximate.covariance_matrix,
        exact.covariance_matrix,
        rtol=3e-4,
    )


def test_mean_trajectory_linearization_runs_for_nonlinear_sde():
    transition = _discretize_state_evolution(
        _nonlinear_model().state_evolution,
        MeanTrajectoryLinearizationConfig(ode_solver=_ode_config()),
    )(jnp.array([0.2]), None, 0.0, 0.1)

    assert jnp.all(jnp.isfinite(transition.mean))
    assert jnp.all(jnp.isfinite(transition.covariance_matrix))


def test_diffrax_sample_transition_samples_but_has_no_density():
    transition = _discretize_state_evolution(
        _nonlinear_model().state_evolution,
        _diffrax_config(),
    )(jnp.array([0.0]), None, 0.0, 0.05)

    first = transition.sample(jr.PRNGKey(1))
    assert transition.has_rsample
    assert jnp.array_equal(first, transition.sample(jr.PRNGKey(1)))
    assert transition.sample(jr.PRNGKey(2), sample_shape=(2,)).shape == (2, 1)
    assert jnp.array_equal(
        transition.rsample(jr.PRNGKey(2), sample_shape=(2,)),
        transition.sample(jr.PRNGKey(2), sample_shape=(2,)),
    )
    with pytest.raises(NotImplementedError, match="sampling only"):
        transition.log_prob(first)


def test_diffrax_rsample_is_differentiable_in_initial_state():
    evolution = _discretize_state_evolution(
        _affine_model().state_evolution,
        _diffrax_config(),
    )
    key = jr.PRNGKey(3)

    def sample_endpoint(initial_value):
        transition = evolution(jnp.array([initial_value]), None, 0.0, 0.05)
        return transition.rsample(key)[0]

    gradient = jax.grad(sample_endpoint)(jnp.array(0.2))
    expected_gradient = (1.0 - 0.7 * 0.01) ** 5

    assert jnp.allclose(gradient, expected_gradient)


@pytest.mark.parametrize(
    "filter_config",
    [
        EnKFConfig(n_particles=4, crn_seed=jr.PRNGKey(3)),
        PFConfig(n_particles=8, crn_seed=jr.PRNGKey(3)),
    ],
)
def test_diffrax_sample_transition_runs_sample_based_filters(filter_config):
    def model(obs_times, obs_values):
        return dsx.sample(
            "f",
            _nonlinear_model(),
            obs_times=obs_times,
            obs_values=obs_values,
        )

    with Filter(filter_config):
        with Discretizer(_diffrax_config()):
            with trace() as tr, seed(rng_seed=2):
                model(jnp.array([0.0, 0.02]), jnp.zeros((2, 1)))

    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
