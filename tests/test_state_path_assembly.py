"""Tests for internal state-path assembly utilities."""

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

import dynestyx as dsx
from dynestyx.inference.latent.parameterization import assemble_state_path


def test_assemble_state_path_discrete_is_identity_on_full_path():
    state_times = jnp.array([0.0, 1.0, 3.0])
    state_path_params = jnp.array([[0.2, -0.1], [0.4, 0.3], [0.9, -0.2]])

    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=dsx.LinearGaussianStateEvolution(
            A=0.8 * jnp.eye(2),
            cov=0.1 * jnp.eye(2),
        ),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.eye(2),
            R=0.2 * jnp.eye(2),
        ),
    )

    assembled = assemble_state_path(
        dynamics,
        state_path_params=state_path_params,
        state_path_param_times=state_times,
    )

    assert jnp.array_equal(assembled.state_path_params, state_path_params)
    assert jnp.array_equal(assembled.state_path, state_path_params)
    assert jnp.array_equal(assembled.state_path_param_times, state_times)
    assert jnp.array_equal(assembled.state_path_times, state_times)


def test_assemble_state_path_ode_reconstructs_path_from_ic():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    x0 = jnp.array(0.3)

    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: 0.0 * x),
        observation_model=lambda x, u, t: dist.Normal(x, 1.0),
    )

    assembled = assemble_state_path(
        dynamics,
        state_path_params=x0,
        state_path_param_times=jnp.array([0.0]),
        obs_times=obs_times,
    )

    assert assembled.state_path_params.shape == (1,)
    assert jnp.allclose(assembled.state_path_params[0], x0)
    assert jnp.array_equal(
        assembled.state_path_times,
        jnp.array([0.0, 0.0, 1.0, 2.0]),
    )
    assert jnp.allclose(assembled.state_path, jnp.full((4,), x0))


def test_assemble_state_path_ode_includes_ic_when_obs_start_later():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: 0.0 * x),
        observation_model=lambda x, u, t: dist.Normal(x, 1.0),
    )

    assembled = assemble_state_path(
        dynamics,
        state_path_params=jnp.array(0.4),
        state_path_param_times=jnp.array([0.0]),
        obs_times=jnp.array([1.0, 2.0]),
    )

    assert jnp.array_equal(assembled.state_path_times, jnp.array([0.0, 1.0, 2.0]))
    assert jnp.allclose(assembled.state_path, jnp.array([0.4, 0.4, 0.4]))


def test_assemble_state_path_sde_requires_discretization():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.2 * x,
            diffusion=dsx.ScalarDiffusion(jnp.array(0.2), bm_dim=1),
        ),
        observation_model=lambda x, u, t: dist.Normal(x, 1.0),
    )

    with pytest.raises(ValueError, match="discretize"):
        assemble_state_path(
            dynamics,
            state_path_params=jnp.array([0.1]),
            state_path_param_times=jnp.array([0.0]),
        )
