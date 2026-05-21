import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.handlers import trace
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Filter, Smoother
from dynestyx.inference.filter_configs import EKFConfig, EnKFConfig, KFConfig
from dynestyx.inference.smoother_configs import EKFSmootherConfig, KFSmootherConfig

_EQX_ERRORS = (
    ValueError,
    eqx.EquinoxRuntimeError,
    eqx.EquinoxTracetimeError,
)


def _identity_lti_model(obs_times=None, obs_values=None, predict_times=None):
    dynamics = dsx.LTI_discrete(
        A=jnp.array([[0.92, 0.08], [0.03, 0.88]]),
        Q=0.02 * jnp.eye(2),
        H=jnp.eye(2),
        R=0.05 * jnp.eye(2),
        initial_mean=jnp.array([0.0, 0.0]),
        initial_cov=0.2 * jnp.eye(2),
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def _simulate_identity_lti_data():
    obs_times = jnp.arange(12.0)
    predictive = Predictive(
        _identity_lti_model,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator(n_simulations=1):
        samples = predictive(jr.PRNGKey(0), predict_times=obs_times)
    return obs_times, samples["f_states"][0, 0], samples["f_observations"][0, 0]


def _partial_missing_observations(obs_values):
    missing = obs_values
    missing = missing.at[2, 0].set(jnp.nan)
    missing = missing.at[3, 1].set(jnp.nan)
    missing = missing.at[7, 0].set(jnp.nan)
    missing = missing.at[9, 1].set(jnp.nan)
    return missing


@pytest.mark.parametrize(
    ("mode", "config", "supported", "mean_site", "error_match"),
    [
        pytest.param(
            "filter",
            KFConfig(
                filter_source="cuthbert",
                record_filtered_states_mean=True,
            ),
            True,
            "f_filtered_states_mean",
            None,
            id="kf-filter",
        ),
        pytest.param(
            "filter",
            EnKFConfig(
                filter_source="cuthbert",
                n_particles=32,
                record_filtered_states_mean=True,
                crn_seed=jr.PRNGKey(1),
            ),
            True,
            "f_filtered_states_mean",
            None,
            id="enkf-filter",
        ),
        pytest.param(
            "filter",
            EKFConfig(
                filter_source="cuthbert",
                record_filtered_states_mean=True,
            ),
            False,
            None,
            "supported only for cuthbert KFConfig and EnKFConfig filters",
            id="ekf-filter",
        ),
        pytest.param(
            "filter",
            KFConfig(
                filter_source="cd_dynamax",
                record_filtered_states_mean=True,
            ),
            False,
            None,
            "CD-Dynamax filters do not support NaNs in obs_values",
            id="cd-dynamax-filter",
        ),
        pytest.param(
            "smoother",
            KFSmootherConfig(
                filter_source="cuthbert",
                record_smoothed_states_mean=True,
            ),
            True,
            "f_smoothed_states_mean",
            None,
            id="kf-smoother",
        ),
        pytest.param(
            "smoother",
            EKFSmootherConfig(
                filter_source="cuthbert",
                record_smoothed_states_mean=True,
            ),
            False,
            None,
            "supported only for cuthbert KFSmootherConfig smoothers",
            id="ekf-smoother",
        ),
        pytest.param(
            "smoother",
            KFSmootherConfig(
                filter_source="cd_dynamax",
                record_smoothed_states_mean=True,
            ),
            False,
            None,
            "CD-Dynamax smoothers do not support NaNs in obs_values",
            id="cd-dynamax-smoother",
        ),
    ],
)
def test_cuthbert_gaussian_discrete_time_missing_observation_support_matrix(
    mode, config, supported, mean_site, error_match
):
    obs_times, true_states, obs_values = _simulate_identity_lti_data()
    missing_obs_values = _partial_missing_observations(obs_values)

    context = (
        Filter(filter_config=config)
        if mode == "filter"
        else Smoother(smoother_config=config)
    )
    if not supported:
        with pytest.raises(_EQX_ERRORS, match=error_match):
            with context:
                _identity_lti_model(obs_times=obs_times, obs_values=missing_obs_values)
        return

    with trace() as tr:
        with context:
            _identity_lti_model(obs_times=obs_times, obs_values=missing_obs_values)

    state_mean = tr[mean_site]["value"]
    assert state_mean.shape == true_states.shape
    assert not jnp.isnan(tr["f_marginal_loglik"]["value"])
    assert not jnp.isnan(state_mean).any()
