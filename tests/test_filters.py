import jax.numpy as jnp
import jax.random as jr
from numpyro.handlers import seed, trace

from tests.fixtures import (
    data_conditioned_jumpy_controls,
    data_conditioned_jumpy_controls_ode,
    data_conditioned_jumpy_controls_sde,
)


def test_jumpy_controls():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic["observations"][0, ...]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 1e-1


def test_jumpy_controls_sde():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_sde()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic["observations"][0, ...]
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 1e-2


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
