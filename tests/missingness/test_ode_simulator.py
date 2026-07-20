import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS, Predictive

from dynestyx import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    GaussianObservation,
    LatentPathBuilder,
    LinearGaussianObservation,
    ODESimulator,
)
from tests.missingness.models import (
    GAUSSIAN_R,
    INDEPENDENT_SCALE,
    ODE_A,
    _independent_observation_mean,
    _nonlinear_observation_mean,
    ode_independent_normal_model,
    ode_linear_gaussian_model,
    ode_nonlinear_gaussian_model,
    sampled_ode_linear_gaussian_model,
)
from tests.missingness.utils import (
    manual_masked_independent_normal_log_prob,
    manual_masked_mvn_log_prob,
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


def _run_ode_latent_trace(model, *, obs_times, obs_values, conditioned_data=None):
    model_to_run = (
        model if conditioned_data is None else condition(model, data=conditioned_data)
    )
    with LatentPathBuilder():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model_to_run(
                obs_times=obs_times,
                obs_values=obs_values,
            )
    return tr


def _ode_initial_condition_from_forward(trace):
    return jnp.asarray(trace["f_states"]["value"])[0, 0]


def _ode_states_at_obs_times(trace, obs_times):
    state_path = jnp.asarray(trace["f_state_path"]["value"])
    state_path_times = jnp.asarray(trace["f_state_path_times"]["value"])
    obs_indices = jnp.searchsorted(
        state_path_times, jnp.asarray(obs_times), side="left"
    )
    return state_path[obs_indices]


def _make_ode_linear_gaussian_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: ODE_A @ x),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
    )


def _make_ode_nonlinear_gaussian_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: ODE_A @ x),
        observation_model=GaussianObservation(
            h=_nonlinear_observation_mean, R=GAUSSIAN_R
        ),
    )


def _make_ode_independent_normal_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: ODE_A @ x),
        observation_model=lambda x, u, t: dist.Independent(
            dist.Normal(_independent_observation_mean(x, u, t), INDEPENDENT_SCALE), 1
        ),
    )


def test_ode_no_missing_conditioning_uses_log_prob_path():
    times = jnp.linspace(0.0, 0.4, 5)
    forward = _run_ode_trace(ode_linear_gaussian_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    initial_condition = _ode_initial_condition_from_forward(forward)
    conditioned = _run_ode_latent_trace(
        ode_linear_gaussian_model,
        obs_times=times,
        obs_values=obs_values,
        conditioned_data={"f_state_path_params": initial_condition},
    )

    states = _ode_states_at_obs_times(conditioned, times)
    expected_observation_log_prob = jnp.sum(
        jnp.stack(
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
    )
    expected_joint_log_prob = (
        _make_ode_linear_gaussian_dynamics().initial_condition.log_prob(
            conditioned["f_state_path"]["value"][0]
        )
        + expected_observation_log_prob
    )
    assert jnp.allclose(conditioned["f_state_path_params"]["value"], initial_condition)
    assert jnp.allclose(
        conditioned["f_joint_log_prob"]["value"], expected_joint_log_prob
    )


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
    initial_condition = _ode_initial_condition_from_forward(forward)
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=1)

    conditioned = _run_ode_latent_trace(
        model,
        obs_times=times,
        obs_values=obs_values,
        conditioned_data={"f_state_path_params": initial_condition},
    )

    states = _ode_states_at_obs_times(conditioned, times)
    expected = []
    for k in range(len(times)):
        mask = jnp.isfinite(obs_values[k])
        safe_obs = jnp.where(mask, obs_values[k], 0.0)
        mu = mean_fn(states[k], times[k])
        expected.append(manual_masked_mvn_log_prob(mu, GAUSSIAN_R, safe_obs, mask))

    dynamics = (
        _make_ode_linear_gaussian_dynamics()
        if model is ode_linear_gaussian_model
        else _make_ode_nonlinear_gaussian_dynamics()
    )
    actual = conditioned["f_joint_log_prob"][
        "value"
    ] - dynamics.initial_condition.log_prob(conditioned["f_state_path"]["value"][0])
    assert jnp.allclose(actual, jnp.sum(jnp.stack(expected)))


def test_ode_independent_missingness_factor_values_match_manual_reference():
    times = jnp.linspace(0.0, 0.4, 5)
    forward = _run_ode_trace(ode_independent_normal_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    initial_condition = _ode_initial_condition_from_forward(forward)
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_partial_row_missing(obs_values, 4, dim_idx=0)

    conditioned = _run_ode_latent_trace(
        ode_independent_normal_model,
        obs_times=times,
        obs_values=obs_values,
        conditioned_data={"f_state_path_params": initial_condition},
    )

    states = _ode_states_at_obs_times(conditioned, times)
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

    actual = conditioned["f_joint_log_prob"][
        "value"
    ] - _make_ode_independent_normal_dynamics().initial_condition.log_prob(
        conditioned["f_state_path"]["value"][0]
    )
    assert jnp.allclose(actual, jnp.sum(jnp.stack(expected)))


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

    with LatentPathBuilder():
        mcmc = MCMC(
            NUTS(sampled_ode_linear_gaussian_model),
            num_samples=1,
            num_warmup=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(2), obs_times=times, obs_values=obs_values)

    assert "alpha" in mcmc.get_samples()
