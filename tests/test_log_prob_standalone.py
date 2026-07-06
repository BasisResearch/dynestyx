"""Tests for the pure-JAX ``dsx.log_prob`` API."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

import dynestyx as dsx
from tests.missingness.utils import manual_masked_mvn_log_prob


def test_log_prob_discrete_matches_manual_joint_density():
    state_times = jnp.array([0.0, 1.0, 2.0])
    state_path_params = jnp.array([0.2, -0.1, 0.4])
    ctrl_times = state_times
    ctrl_values = jnp.array([0.5, -0.25, 0.75])
    obs_times = jnp.array([0.0, 2.0])
    obs_values = jnp.array([0.3, -0.2])

    dynamics = dsx.DynamicalModel(
        control_dim=1,
        initial_condition=dist.Normal(0.0, 1.1),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(0.7 * x + 0.5 * u, 0.3),
        observation_model=lambda x, u, t: dist.Normal(x - 0.25 * u, 0.4),
    )

    actual = dsx.log_prob(
        dynamics,
        state_path_params=state_path_params,
        state_path_param_times=state_times,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    expected = dynamics.initial_condition.log_prob(state_path_params[0])
    expected = expected + dynamics.state_evolution(
        state_path_params[0], ctrl_values[0], state_times[0], state_times[1]
    ).log_prob(state_path_params[1])
    expected = expected + dynamics.state_evolution(
        state_path_params[1], ctrl_values[1], state_times[1], state_times[2]
    ).log_prob(state_path_params[2])
    expected = expected + dynamics.observation_model(
        state_path_params[0], ctrl_values[0], obs_times[0]
    ).log_prob(obs_values[0])
    expected = expected + dynamics.observation_model(
        state_path_params[2], ctrl_values[2], obs_times[1]
    ).log_prob(obs_values[1])

    assert jnp.allclose(actual, expected)
    assert jnp.allclose(
        actual,
        dsx.log_prob(
            dynamics,
            state_path_params=state_path_params,
            state_path_param_times=state_times,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            chunk_size=1,
        ),
    )


def test_log_prob_discrete_missingness_matches_manual_masking():
    state_times = jnp.array([0.0, 1.0])
    state_path_params = jnp.array([[0.2, -0.1], [0.4, 0.3]])
    obs_values = jnp.array([[0.15, jnp.nan], [0.35, 0.5]])
    cov = jnp.array([[0.3, 0.05], [0.05, 0.25]])

    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=dsx.LinearGaussianStateEvolution(
            A=0.8 * jnp.eye(2),
            cov=0.1 * jnp.eye(2),
        ),
        observation_model=dsx.LinearGaussianObservation(H=jnp.eye(2), R=cov),
    )

    actual = dsx.log_prob(
        dynamics,
        state_path_params=state_path_params,
        state_path_param_times=state_times,
        obs_times=state_times,
        obs_values=obs_values,
    )

    expected = dynamics.initial_condition.log_prob(state_path_params[0])
    expected = expected + dynamics.state_evolution(
        state_path_params[0], None, state_times[0], state_times[1]
    ).log_prob(state_path_params[1])
    expected = expected + manual_masked_mvn_log_prob(
        state_path_params[0],
        cov,
        jnp.array([0.15, 0.0]),
        jnp.array([True, False]),
    )
    expected = expected + manual_masked_mvn_log_prob(
        state_path_params[1],
        cov,
        obs_values[1],
        jnp.array([True, True]),
    )

    assert jnp.allclose(actual, expected)


def test_log_prob_ode_uses_initial_condition_as_only_latent():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.1, -0.2, 0.3])

    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 0.7),
        state_evolution=dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: 0.0 * x),
        observation_model=lambda x, u, t: dist.Normal(x, 0.25),
    )

    actual = dsx.log_prob(
        dynamics,
        state_path_params=jnp.array(0.2),
        state_path_param_times=jnp.array([0.0]),
        obs_times=obs_times,
        obs_values=obs_values,
    )

    expected = dynamics.initial_condition.log_prob(jnp.array(0.2))
    expected = expected + jnp.sum(dist.Normal(0.2, 0.25).log_prob(obs_values))
    assert jnp.allclose(actual, expected)


def test_log_prob_respects_explicit_t0():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        t0=1.0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(x, 1.0),
        observation_model=lambda x, u, t: dist.Normal(x, 1.0),
    )

    with pytest.raises(Exception, match="dynamics.t0"):
        dsx.log_prob(
            dynamics,
            state_path_params=jnp.array([0.1, 0.2]),
            state_path_param_times=jnp.array([0.0, 1.0]),
        )


def test_log_prob_ode_rejects_multiple_latent_times():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.1 * x
        ),
        observation_model=lambda x, u, t: dist.Normal(x, 1.0),
    )

    with pytest.raises(ValueError, match="exactly one latent path parameter"):
        dsx.log_prob(
            dynamics,
            state_path_params=jnp.array([0.1, 0.2]),
            state_path_param_times=jnp.array([0.0, 1.0]),
        )


def test_log_prob_sde_requires_discretization():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.1 * x,
            diffusion=dsx.ScalarDiffusion(jnp.array(0.2), bm_dim=1),
        ),
        observation_model=lambda x, u, t: dist.Normal(x, 1.0),
    )

    with pytest.raises(ValueError, match="discretize"):
        dsx.log_prob(
            dynamics,
            state_path_params=jnp.array([0.1]),
            state_path_param_times=jnp.array([0.0]),
        )
