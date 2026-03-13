import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.handlers import seed, trace

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

    synthetic_observations = synthetic["observations"][0, ...]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < mean_error_tol


def test_jumpy_controls_sde():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_sde()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic["observations"][0, ...]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 2e-2


def test_jumpy_controls_ode():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_ode()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic["observations"][0, ...]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 0.1
