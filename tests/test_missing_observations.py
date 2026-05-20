import equinox as eqx
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import trace
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Filter, Smoother
from dynestyx.inference.filter_configs import EKFConfig, EnKFConfig, KFConfig
from dynestyx.inference.smoother_configs import EKFSmootherConfig, KFSmootherConfig
from tests.models import jumpy_controls_model

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


def _block_missing_observations(obs_values):
    return obs_values.at[4:8, :].set(jnp.nan)


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


def test_sample_rejects_nan_obs_times():
    obs_times = jnp.array([0.0, jnp.nan, 2.0])
    obs_values = jnp.zeros((3, 1))
    ctrl_times = jnp.array([0.0, 1.0, 2.0])
    ctrl_values = jnp.zeros((3, 1))

    with pytest.raises(
        _EQX_ERRORS,
        match="obs_times must not contain NaNs",
    ):
        jumpy_controls_model(
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def test_sample_rejects_nan_ctrl_times():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.zeros((3, 1))
    ctrl_times = jnp.array([0.0, jnp.nan, 2.0])
    ctrl_values = jnp.zeros((3, 1))

    with pytest.raises(
        _EQX_ERRORS,
        match="ctrl_times must not contain NaNs",
    ):
        jumpy_controls_model(
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def test_sample_rejects_nan_predict_times():
    predict_times = jnp.array([0.0, jnp.nan, 2.0])

    with pytest.raises(
        _EQX_ERRORS,
        match="predict_times must not contain NaNs",
    ):
        jumpy_controls_model(predict_times=predict_times)


def test_sample_rejects_nan_ctrl_values():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.zeros((3, 1))
    ctrl_times = obs_times
    ctrl_values = jnp.array([[0.0], [jnp.nan], [1.0]])

    with pytest.raises(
        _EQX_ERRORS,
        match="ctrl_values must not contain NaNs",
    ):
        jumpy_controls_model(
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def test_cd_dynamax_filter_rejects_missing_observations():
    obs_times, _, obs_values = _simulate_identity_lti_data()
    missing_obs_values = _block_missing_observations(obs_values)

    with pytest.raises(
        _EQX_ERRORS,
        match="CD-Dynamax filters do not support NaNs in obs_values",
    ):
        with Filter(
            filter_config=KFConfig(
                filter_source="cd_dynamax",
                record_filtered_states_mean=True,
            )
        ):
            _identity_lti_model(obs_times=obs_times, obs_values=missing_obs_values)


def test_cd_dynamax_smoother_rejects_missing_observations():
    obs_times, _, obs_values = _simulate_identity_lti_data()
    missing_obs_values = _block_missing_observations(obs_values)

    with pytest.raises(
        _EQX_ERRORS,
        match="CD-Dynamax smoothers do not support NaNs in obs_values",
    ):
        with Smoother(
            smoother_config=KFSmootherConfig(
                filter_source="cd_dynamax",
                record_smoothed_states_mean=True,
            )
        ):
            _identity_lti_model(obs_times=obs_times, obs_values=missing_obs_values)


def test_cuthbert_kf_plate_missingness_hierarchical_pooling_smoke():
    m = 4
    obs_times = jnp.arange(0.0, 20.0, 1.0)
    a_base = jnp.array([[0.0, 0.2], [-0.1, 0.8]])
    q = jnp.array([[0.1, 0.01], [-0.01, 0.15]])
    h = jnp.eye(2)
    r = jnp.diag(jnp.array([0.25, 0.25]))
    alpha_offset = 0.15
    alpha_scale = 0.8

    def raw_to_alpha(raw):
        return alpha_offset + alpha_scale * jnn.sigmoid(raw)

    def hierarchical_missing_model(
        obs_times=None, obs_values=None, predict_times=None, m=m
    ):
        mu_alpha_raw = numpyro.sample("mu_alpha_raw", dist.Normal(0.0, 1.0))
        sigma_alpha_raw = numpyro.sample("sigma_alpha_raw", dist.HalfNormal(0.4))

        with dsx.plate("trajectories", m):
            alpha_raw = numpyro.sample(
                "alpha_raw", dist.Normal(mu_alpha_raw, sigma_alpha_raw)
            )
            alpha = raw_to_alpha(alpha_raw)
            numpyro.deterministic("alpha", alpha)
            a = jnp.repeat(a_base[None], m, axis=0).at[:, 0, 0].set(alpha)
            dynamics = dsx.LTI_discrete(A=a, Q=q, H=h, R=r)
            return dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    true_params = {
        "mu_alpha_raw": jnp.array(0.35),
        "sigma_alpha_raw": jnp.array(0.14),
    }

    with DiscreteTimeSimulator():
        synthetic = Predictive(
            hierarchical_missing_model,
            params=true_params,
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(0), predict_times=obs_times, m=m)

    obs_values = np.array(synthetic["f_observations"][0, :, 0], copy=True)
    alpha_true = np.asarray(synthetic["alpha"][0])

    obs_values[0, 5:10, :] = np.nan
    obs_values[1, [4, 5, 12, 13, 17], 0] = np.nan
    obs_values[2, [3, 9, 15], 1] = np.nan
    obs_values[3, 4:16, :] = np.nan
    obs_values = jnp.array(obs_values)

    def conditioned_filter(obs_times=None, obs_values=None, m=m):
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
            hierarchical_missing_model(
                obs_times=obs_times,
                obs_values=obs_values,
                m=m,
            )

    mcmc = MCMC(
        NUTS(conditioned_filter),
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(1), obs_times=obs_times, obs_values=obs_values, m=m)

    posterior = mcmc.get_samples()
    assert posterior["mu_alpha_raw"].shape == (10,)
    assert posterior["sigma_alpha_raw"].shape == (10,)
    assert posterior["alpha_raw"].shape == (10, m)
    assert posterior["alpha"].shape == (10, m)
    assert not jnp.isnan(posterior["mu_alpha_raw"]).any()
    assert not jnp.isnan(posterior["sigma_alpha_raw"]).any()
    assert not jnp.isnan(posterior["alpha_raw"]).any()
    assert not jnp.isnan(posterior["alpha"]).any()

    alpha_post = np.asarray(posterior["alpha"])
    alpha_mean = alpha_post.mean(axis=0)
    mean_abs_error = np.abs(alpha_mean - alpha_true).mean()
    sparse_traj_error = abs(alpha_mean[3] - alpha_true[3])

    assert mean_abs_error < 0.2
    assert sparse_traj_error < 0.25
