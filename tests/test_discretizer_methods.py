"""Minimal tests for the configuration-driven SDE discretizers."""

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
    _discretize_state_evolution,
)
from dynestyx.inference.configs.filter import EnKFConfig, PFConfig
from dynestyx.inference.configs.simulator import (
    ODESimulatorConfig,
    SDESimulatorConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    LinearGaussianObservation,
)


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
