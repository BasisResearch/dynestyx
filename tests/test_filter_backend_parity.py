"""Regression tests for backend forecast, key, and resampling conventions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cd_dynamax import ContDiscreteNonlinearGaussianSSM
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import (
    PosteriorCDNLSSMFiltered,
)

import dynestyx as dsx
from dynestyx import configs
from dynestyx.inference.custom_filter import run_filter_steps, step_inputs
from dynestyx.inference.integrations.cd_dynamax.utils import dsx_to_cd_dynamax
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    build_cuthbert_filter,
)

TIMES = jnp.array([0.0, 0.1, 0.25, 0.3])
OBSERVATIONS = jnp.array([[0.1], [-0.5], [0.6], [0.3]])
CONTROLS = jnp.array([[0.2], [-0.3], [0.4], [-0.5]])
KEY = jax.random.PRNGKey(42)


def check_posterior(model, config, expected):
    with dsx.Filter(config):
        result = dsx.condition(
            "x",
            model,
            obs_times=TIMES,
            obs_values=OBSERVATIONS,
            ctrl_times=TIMES,
            ctrl_values=CONTROLS,
        )
    assert jax.tree.structure(result.states) == jax.tree.structure(expected)
    for (path, ref), got in zip(
        jax.tree.leaves_with_path(expected), jax.tree.leaves(result.states), strict=True
    ):
        np.testing.assert_allclose(got, ref, atol=2e-5, rtol=1e-4, err_msg=str(path))


@pytest.fixture
def model():
    return dsx.LTI_discrete(
        A=jnp.array([[0.9]]),
        Q=0.1 * jnp.eye(1),
        H=jnp.eye(1),
        R=0.2 * jnp.eye(1),
        B=jnp.eye(1),
    )


class CustomUKF(configs.SplitFilterConfig):
    def build_filter(self, model):
        return configs.UKFConfig().build_filter(model)

    def gather_outputs(self, prior, filtered):
        return filtered


@pytest.mark.parametrize(
    "config", [configs.EKFConfig(filter_source="cd_dynamax"), CustomUKF()]
)
def test_discrete_forecasts_include_the_terminal_step(config, model):
    with dsx.Filter(config):
        result = dsx.condition(
            "x",
            model,
            obs_times=TIMES,
            obs_values=OBSERVATIONS,
            ctrl_times=TIMES,
            ctrl_values=CONTROLS,
        )
    posterior = result.states
    assert isinstance(posterior, PosteriorGSSMFiltered)
    assert (
        posterior.filtered_means is not None
        and posterior.filtered_covariances is not None
    )
    np.testing.assert_allclose(
        posterior.predicted_means, 0.9 * posterior.filtered_means + CONTROLS, atol=2e-6
    )
    np.testing.assert_allclose(
        posterior.predicted_covariances,
        0.9**2 * posterior.filtered_covariances + 0.1,
        atol=2e-6,
    )


def test_cd_enkf_preserves_seeded_forecast_and_analysis_schedule():
    model = dsx.LTI_continuous(
        A=-jnp.eye(1),
        L=0.2 * jnp.eye(1),
        H=jnp.eye(1),
        R=0.2 * jnp.eye(1),
        B=jnp.eye(1),
    )
    config = configs.ContinuousTimeEnKFConfig(n_particles=8, crn_seed=KEY)
    backend = ContDiscreteNonlinearGaussianSSM(1, 1, 1)
    params, _ = dsx_to_cd_dynamax(model, cd_model=backend)
    expected = backend.filter(
        params,
        emissions=OBSERVATIONS,
        t_emissions=TIMES[:, None],
        inputs=CONTROLS,
        filter_type="EnKF",
        enkf_N_particles=config.n_particles,
        enkf_inflation_delta=config.inflation_delta,
        key=KEY,
        filter_state_order=config.filter_state_order,
        diffeqsolve_dt0=config.diffeqsolve_dt0,
        diffeqsolve_max_steps=config.diffeqsolve_max_steps,
        output_fields=[
            "marginal_loglik",
            "filtered_means",
            "filtered_covariances",
            "y_pred_mean",
            "y_pred_cov",
            "y_obs_pred_mean",
            "y_obs_pred_cov",
            "y_ens_pred",
            "y_obs_ens_pred",
        ],
    )
    check_posterior(model, config, expected)


def test_cd_particle_history_retains_weights_before_resampling():
    model = dsx.LTI_continuous(
        A=-jnp.eye(1),
        L=0.2 * jnp.eye(1),
        H=jnp.eye(1),
        R=0.2 * jnp.eye(1),
    )
    config = configs.ContinuousTimeDPFConfig(
        n_particles=8, ess_threshold_ratio=1.0, crn_seed=KEY
    )
    with dsx.Filter(config):
        result = dsx.condition(
            "x", model, obs_times=TIMES[:1], obs_values=OBSERVATIONS[:1]
        )
    posterior = result.states
    assert isinstance(posterior, PosteriorCDNLSSMFiltered)
    assert posterior.particles is not None
    assert posterior.log_weights is not None
    observation_loglik = (
        -0.5 * (OBSERVATIONS[0, 0] - posterior.particles[0, :, 0]) ** 2 / 0.2
    )
    np.testing.assert_allclose(
        posterior.log_weights[0], jax.nn.log_softmax(observation_loglik), atol=2e-6
    )


def test_online_particle_overrides(model):
    pf, parallel = build_cuthbert_filter(
        model,
        configs.PFConfig(),
        KEY,
        want_parallel=False,
        extra_filter_kwargs=dict(
            n_filter_particles=12,
            resampling_base_method="multinomial",
            ess_threshold=1.0,
        ),
    )
    inputs = step_inputs(model, TIMES, OBSERVATIONS, TIMES, CONTROLS)
    inputs = inputs._replace(
        time_prev=inputs.time_prev.at[0].set(2 * TIMES[0] - TIMES[1])
    )
    states = run_filter_steps(pf, inputs, KEY, lambda state: state, parallel=parallel)
    config = configs.PFConfig(
        crn_seed=KEY,
        n_particles=12,
        ess_threshold_ratio=1.0,
        resampling_method=configs.PFResamplingConfig(base_method="multinomial"),
    )
    check_posterior(model, config, states)
