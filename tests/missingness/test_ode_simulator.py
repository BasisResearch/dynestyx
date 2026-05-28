import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS, Predictive

from dynestyx import ODESimulator
from tests.missingness.models import (
    GAUSSIAN_R,
    INDEPENDENT_SCALE,
    _independent_observation_mean,
    _nonlinear_observation_mean,
    ode_independent_normal_model,
    ode_linear_gaussian_model,
    ode_nonlinear_gaussian_model,
    sampled_ode_linear_gaussian_model,
)
from tests.missingness.utils import (
    latent_conditioning_data,
    manual_masked_independent_normal_log_prob,
    manual_masked_mvn_log_prob,
    observation_log_probs,
    set_full_row_missing,
    set_partial_row_missing,
)


def _run_ode_trace(model, *, obs_times=None, obs_values=None, predict_times=None):
    with ODESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )
    return tr


def test_ode_no_missing_path_preserves_observation_sample_sites():
    times = jnp.linspace(0.0, 0.4, 5)
    forward = _run_ode_trace(ode_linear_gaussian_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]

    conditioned = _run_ode_trace(
        ode_linear_gaussian_model, obs_times=times, obs_values=obs_values
    )

    assert observation_log_probs(conditioned).shape == (len(times),)


@pytest.mark.parametrize(
    ("model", "mean_fn"),
    [
        (ode_linear_gaussian_model, lambda x, t: x),
        (
            ode_nonlinear_gaussian_model,
            lambda x, t: _nonlinear_observation_mean(x, None, t),
        ),
    ],
)
def test_ode_gaussian_missingness_factor_values_match_manual_reference(
    model,
    mean_fn,
):
    times = jnp.linspace(0.0, 0.4, 5)
    forward = _run_ode_trace(model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    latent_data = latent_conditioning_data(forward)
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=1)

    conditioned_model = condition(model, data=latent_data)
    conditioned = _run_ode_trace(
        conditioned_model, obs_times=times, obs_values=obs_values
    )

    states = conditioned["f_states"]["value"][0]
    observations = conditioned["f_observations"]["value"][0]
    assert states.shape == (len(times), 2)
    assert jnp.array_equal(jnp.isnan(observations), jnp.isnan(obs_values))
    assert jnp.allclose(jnp.nan_to_num(observations), jnp.nan_to_num(obs_values))

    expected = []
    for k in range(len(times)):
        mask = jnp.isfinite(obs_values[k])
        safe_obs = jnp.where(mask, obs_values[k], 0.0)
        mu = mean_fn(states[k], times[k])
        expected.append(manual_masked_mvn_log_prob(mu, GAUSSIAN_R, safe_obs, mask))

    actual = observation_log_probs(conditioned)
    assert jnp.allclose(actual, jnp.stack(expected))


def test_ode_independent_missingness_factor_values_match_manual_reference():
    times = jnp.linspace(0.0, 0.4, 5)
    forward = _run_ode_trace(ode_independent_normal_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    latent_data = latent_conditioning_data(forward)
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_partial_row_missing(obs_values, 4, dim_idx=0)

    conditioned_model = condition(ode_independent_normal_model, data=latent_data)
    conditioned = _run_ode_trace(
        conditioned_model, obs_times=times, obs_values=obs_values
    )

    states = conditioned["f_states"]["value"][0]
    observations = conditioned["f_observations"]["value"][0]
    assert jnp.array_equal(jnp.isnan(observations), jnp.isnan(obs_values))
    assert jnp.allclose(jnp.nan_to_num(observations), jnp.nan_to_num(obs_values))

    expected = []
    for k in range(len(times)):
        mask = jnp.isfinite(obs_values[k])
        safe_obs = jnp.where(mask, obs_values[k], 0.0)
        loc = _independent_observation_mean(states[k], None, times[k])
        expected.append(
            manual_masked_independent_normal_log_prob(
                loc, INDEPENDENT_SCALE, safe_obs, mask
            )
        )

    actual = observation_log_probs(conditioned)
    assert jnp.allclose(actual, jnp.stack(expected))


def test_ode_missingness_mcmc_smoke():
    times = jnp.linspace(0.0, 0.4, 5)
    predictive = Predictive(
        sampled_ode_linear_gaussian_model,
        params={"alpha": jnp.array(0.2)},
        num_samples=1,
        exclude_deterministic=False,
    )
    with ODESimulator():
        generated = predictive(jr.PRNGKey(1), predict_times=times)
    obs_values = generated["f_observations"][0, 0]
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=1)

    with ODESimulator():
        mcmc = MCMC(
            NUTS(sampled_ode_linear_gaussian_model),
            num_samples=1,
            num_warmup=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(2), obs_times=times, obs_values=obs_values)

    assert "alpha" in mcmc.get_samples()
