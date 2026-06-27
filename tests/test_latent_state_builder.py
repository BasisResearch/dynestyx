import jax.numpy as jnp
import jax.random as jr
from numpyro.handlers import condition, seed, trace

from dynestyx import DiscreteTimeSimulator, LatentStateBuilder
from tests.missingness.models import (
    GAUSSIAN_R,
    discrete_dirac_model,
    discrete_linear_gaussian_model,
)
from tests.missingness.utils import (
    latent_conditioning_data,
    manual_masked_mvn_log_prob,
    observation_log_probs,
    set_full_row_missing,
    set_partial_row_missing,
)


def _run_forward_trace(model, *, predict_times):
    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model(predict_times=predict_times)
    return tr


def test_latent_state_builder_dirac_only_frees_missing_rows():
    times = jnp.arange(6.0)
    forward = _run_forward_trace(discrete_dirac_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_full_row_missing(obs_values, 4)

    with LatentStateBuilder():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(1)):
            discrete_dirac_model(obs_times=times, obs_values=obs_values)

    assert tr["f_x_1"]["is_observed"] is False
    assert tr["f_x_4"]["is_observed"] is False
    assert tr["f_x_0"]["is_observed"] is True
    assert tr["f_x_2"]["is_observed"] is True
    assert tr["f_x_3"]["is_observed"] is True
    assert tr["f_x_5"]["is_observed"] is True


def test_latent_state_builder_matches_masked_observation_reference():
    times = jnp.arange(5.0)
    forward = _run_forward_trace(discrete_linear_gaussian_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    latent_data = latent_conditioning_data(forward)
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0)

    conditioned_model = condition(discrete_linear_gaussian_model, data=latent_data)
    with LatentStateBuilder():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(2)):
            conditioned_model(obs_times=times, obs_values=obs_values)

    states = forward["f_states"]["value"][0]
    expected = []
    for k in range(len(times)):
        mask = jnp.isfinite(obs_values[k])
        safe_obs = jnp.where(mask, obs_values[k], 0.0)
        expected.append(
            manual_masked_mvn_log_prob(states[k], GAUSSIAN_R, safe_obs, mask)
        )

    actual = observation_log_probs(tr)
    assert jnp.allclose(actual, jnp.stack(expected))
