import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.inference.filter_configs import ContinuousTimeDPFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel
from tests.fixtures import (
    data_conditioned_jumpy_controls,
    data_conditioned_jumpy_controls_ode,
    data_conditioned_jumpy_controls_sde,
)


@pytest.mark.parametrize(
    ("filter_type", "filter_source", "mean_error_tol"),
    [
        ("kf", "cuthbert", 1e-1),
        ("kf", "cd_dynamax", 1e-1),
        ("ekf", "cuthbert", 1e-1),
        ("ekf", "cd_dynamax", 1e-1),
        ("ukf", "cd_dynamax", 1e-1),
        ("pf", "cuthbert", 1e-1),
    ],
)
def test_jumpy_controls(filter_type, filter_source, mean_error_tol):
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls(
        filter_type=filter_type,
        filter_source=filter_source,
    )
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic[
        "observations"
    ]  # (T, obs_dim) after _normalize_synthetic
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < mean_error_tol


def test_jumpy_controls_sde():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_sde()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic[
        "observations"
    ]  # (T, obs_dim) after _normalize_synthetic
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 3e-2


def test_jumpy_controls_ode():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_ode()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic[
        "observations"
    ]  # (T, obs_dim) after _normalize_synthetic
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 0.1


def test_continuous_time_dpf_non_gaussian_observation_smoke():
    obs_times = jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32)
    obs_values = jnp.array([0, 1, 0], dtype=jnp.int32)

    def model():
        bias = numpyro.sample("bias", dist.Normal(0.0, 0.5))
        dynamics = DynamicalModel(
            initial_condition=dist.LogNormal(loc=jnp.zeros(1), scale=jnp.ones(1)),
            state_evolution=ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -0.3 * jnp.sin(x),
                diffusion_coefficient=lambda x, u, t: 0.1 * jnp.eye(1),
            ),
            observation_model=lambda x, u, t: dist.Poisson(rate=jnp.exp(x[0] + bias)),
        )
        dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    with seed(rng_seed=jr.PRNGKey(0)):
        with Filter(filter_config=ContinuousTimeDPFConfig(n_particles=32)):
            model()
