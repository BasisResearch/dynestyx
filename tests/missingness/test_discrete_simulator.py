import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx import (
    DiscreteTimeSimulator,
    DynamicalModel,
    LatentStateBuilder,
    LinearGaussianStateEvolution,
)
from tests.missingness.models import (
    GAUSSIAN_R,
    INDEPENDENT_SCALE,
    _independent_observation_mean,
    _nonlinear_observation_mean,
    discrete_dirac_model,
    discrete_independent_normal_model,
    discrete_linear_gaussian_model,
    discrete_nonlinear_gaussian_model,
    sampled_discrete_linear_gaussian_model,
)
from tests.missingness.utils import (
    latent_conditioning_data,
    manual_masked_independent_normal_log_prob,
    manual_masked_mvn_log_prob,
    observation_log_probs,
    observation_site_names,
    set_full_row_missing,
    set_partial_row_missing,
)


def _run_discrete_trace(model, *, obs_times=None, obs_values=None, predict_times=None):
    context = (
        LatentStateBuilder()
        if obs_times is not None or obs_values is not None
        else DiscreteTimeSimulator()
    )
    with context:
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )
    return tr


def _correlated_student_t_model(
    alpha=None,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    alpha = numpyro.sample("alpha", dist.Uniform(-0.7, 0.7), obs=alpha)
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=LinearGaussianStateEvolution(
            A=jnp.array([[alpha, 0.2], [-0.1, 0.8]]),
            cov=0.05 * jnp.eye(2),
        ),
        observation_model=lambda x, u, t: dist.MultivariateStudentT(
            df=5.0,
            loc=x,
            scale_tril=jnp.array([[0.4, 0.0], [0.15, 0.5]]),
        ),
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def _scalar_categorical_hmm_like_model(
    A=None,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    A = numpyro.sample(
        "A",
        dist.Dirichlet(jnp.ones(2)).expand([2]).to_event(1),
        obs=A,
    )

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

    def observation_model(x, u, t):
        probs = jnp.array(
            [
                [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2],
            ]
        )
        return dist.Categorical(probs=probs[x])

    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.Categorical(probs=jnp.ones(2) / 2),
        state_evolution=state_evolution,
        observation_model=observation_model,
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def test_discrete_no_missing_conditioning_uses_log_prob_path():
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(discrete_linear_gaussian_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    latent_data = latent_conditioning_data(forward)

    conditioned_model = condition(discrete_linear_gaussian_model, data=latent_data)
    conditioned = _run_discrete_trace(
        conditioned_model, obs_times=times, obs_values=obs_values
    )

    actual = observation_log_probs(conditioned)
    states = conditioned["f_states"]["value"][0]
    expected = jnp.stack(
        [
            manual_masked_mvn_log_prob(
                states[k],
                GAUSSIAN_R,
                obs_values[k],
                jnp.ones_like(obs_values[k], dtype=bool),
            )
            for k in range(len(times))
        ]
    )
    y_sites = observation_site_names(conditioned)
    assert y_sites == [f"f_y_{k}" for k in range(len(times))]
    for k, site_name in enumerate(y_sites):
        assert conditioned[site_name]["type"] == "deterministic"
        assert jnp.allclose(conditioned[site_name]["value"], obs_values[k])
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize(
    ("model", "mean_fn"),
    [
        (discrete_linear_gaussian_model, lambda x, t: x),
        (
            discrete_nonlinear_gaussian_model,
            lambda x, t: _nonlinear_observation_mean(x, None, t),
        ),
    ],
)
def test_discrete_gaussian_missingness_factor_values_match_manual_reference(
    model,
    mean_fn,
):
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    latent_data = latent_conditioning_data(forward)
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0)

    conditioned_model = condition(model, data=latent_data)
    conditioned = _run_discrete_trace(
        conditioned_model, obs_times=times, obs_values=obs_values
    )

    states = conditioned["f_states"]["value"][0]
    observations = conditioned["f_observations"]["value"][0]
    assert states.shape == (len(times), 2)
    assert observations.shape == (len(times), 2)
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


def test_discrete_independent_missingness_factor_values_match_manual_reference():
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(
        discrete_independent_normal_model, predict_times=times
    )
    obs_values = forward["f_observations"]["value"][0]
    latent_data = latent_conditioning_data(forward)
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_partial_row_missing(obs_values, 4, dim_idx=1)

    conditioned_model = condition(discrete_independent_normal_model, data=latent_data)
    conditioned = _run_discrete_trace(
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


def test_discrete_missingness_mcmc_smoke():
    times = jnp.arange(5.0)
    predictive = Predictive(
        sampled_discrete_linear_gaussian_model,
        params={"alpha": jnp.array(0.72)},
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        generated = predictive(jr.PRNGKey(1), predict_times=times)
    obs_values = generated["f_observations"][0, 0]
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0)

    with LatentStateBuilder():
        mcmc = MCMC(
            NUTS(sampled_discrete_linear_gaussian_model),
            num_samples=1,
            num_warmup=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(2), obs_times=times, obs_values=obs_values)

    assert "alpha" in mcmc.get_samples()


def test_discrete_full_row_missing_correlated_student_t_mcmc_smoke():
    times = jnp.arange(8.0)
    with DiscreteTimeSimulator():
        generated = Predictive(
            _correlated_student_t_model,
            params={"alpha": jnp.array(0.3)},
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(3), predict_times=times)
    obs_values = generated["f_observations"][0, 0]
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_full_row_missing(obs_values, 3)
    obs_values = set_full_row_missing(obs_values, 4)

    with LatentStateBuilder():
        mcmc = MCMC(
            NUTS(_correlated_student_t_model),
            num_samples=1,
            num_warmup=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(4), obs_times=times, obs_values=obs_values)

    assert "alpha" in mcmc.get_samples()


def test_discrete_categorical_conditioning_raises_clear_error():
    times = jnp.arange(6.0)
    true_A = jnp.array([[0.95, 0.05], [0.1, 0.9]])
    with DiscreteTimeSimulator():
        generated = Predictive(_scalar_categorical_hmm_like_model, num_samples=1)(
            jr.PRNGKey(5),
            A=true_A,
            predict_times=times,
        )
    obs_values = jnp.asarray(generated["f_observations"])[0, 0, :, 0]

    with pytest.raises(
        ValueError,
        match="Simulator handlers are generation-only",
    ):
        with DiscreteTimeSimulator():
            with seed(rng_seed=jr.PRNGKey(6)):
                _scalar_categorical_hmm_like_model(
                    A=true_A,
                    obs_times=times,
                    obs_values=obs_values,
                )


def test_discrete_dirac_missingness_uses_latent_state_builder_path():
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(discrete_dirac_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    obs_values = set_full_row_missing(obs_values, 2)

    conditioned = _run_discrete_trace(
        discrete_dirac_model, obs_times=times, obs_values=obs_values
    )
    assert "f_x_2" in conditioned
    assert conditioned["f_x_2"]["is_observed"] is False
