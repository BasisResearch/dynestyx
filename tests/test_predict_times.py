import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.inference.filter_configs import ContinuousTimeEKFConfig, EKFConfig, KFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models.lti_dynamics import LTI_continuous, LTI_discrete
from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel, LinearGaussianObservation
from dynestyx.simulators import DiscreteTimeSimulator, ODESimulator, SDESimulator
from tests.test_utils import get_output_dir

SAVE_PREDICTION_PLOTS = True


def _assert_all_finite(*arrays):
    for arr in arrays:
        assert jnp.isfinite(jnp.asarray(arr)).all()


def _discrete_lti_model(obs_times=None, obs_values=None, predict_times=None):
    dynamics = LTI_discrete(
        A=jnp.array([[0.8]]),
        Q=jnp.array([[0.1]]),
        H=jnp.array([[1.0]]),
        R=jnp.array([[0.2]]),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def _continuous_lti_model(obs_times=None, obs_values=None, predict_times=None):
    dynamics = LTI_continuous(
        A=jnp.array([[-1.0]]),
        L=jnp.array([[1.0]]),
        H=jnp.array([[1.0]]),
        R=jnp.array([[0.2]]),
    )
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def test_predict_times_must_be_strictly_after_obs_times_filter():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    predict_times = jnp.array([1.5, 3.0])
    obs_values = jnp.zeros((len(obs_times), 1))

    def model():
        with Filter(filter_config=KFConfig()):
            _discrete_lti_model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    with pytest.raises(ValueError, match="strictly greater than all obs_times"):
        with seed(rng_seed=jr.PRNGKey(0)):
            model()


def test_predict_times_must_be_strictly_after_obs_times_simulator():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    predict_times = jnp.array([2.0, 2.5])

    def model():
        with DiscreteTimeSimulator():
            _discrete_lti_model(obs_times=obs_times, predict_times=predict_times)

    with pytest.raises(ValueError, match="strictly greater than all obs_times"):
        with seed(rng_seed=jr.PRNGKey(0)):
            model()


def test_discrete_kf_filter_emits_forecast_sites():
    obs_times = jnp.arange(0.0, 5.0, 1.0)
    predict_times = jnp.arange(5.0, 8.0, 1.0)
    obs_values = jnp.zeros((len(obs_times), 1))

    def model():
        with Filter(filter_config=KFConfig()):
            _discrete_lti_model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        model()

    assert "f_forecasted_state_means" in tr
    assert "f_forecasted_state_covs" in tr
    means = tr["f_forecasted_state_means"]["value"]
    covs = tr["f_forecasted_state_covs"]["value"]
    assert means.shape[0] == len(predict_times)
    assert covs.shape[0] == len(predict_times)
    _assert_all_finite(means, covs)

    if SAVE_PREDICTION_PLOTS:
        import matplotlib.pyplot as plt

        out = get_output_dir("test_predict_times_discrete_kf")
        t_obs = jnp.asarray(obs_times)
        t_pred = jnp.asarray(predict_times)
        filt = tr["f_filtered_states_mean"]["value"].squeeze(-1)
        pred = means.squeeze(-1)
        y_obs = obs_values.squeeze(-1)

        plt.figure(figsize=(8, 4))
        plt.plot(t_obs, y_obs, "o", label="observations")
        plt.plot(t_obs, filt, "-", label="filtered fit")
        plt.plot(t_pred, pred, "--", label="forecast")
        plt.axvline(float(t_obs[-1]), color="gray", linestyle=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "forecast_window.png", dpi=150, bbox_inches="tight")
        plt.close()


def test_continuous_cd_dynamax_filter_emits_forecast_sites():
    obs_times = jnp.arange(0.0, 1.0, 0.2)
    predict_times = jnp.arange(1.0, 1.6, 0.2)
    obs_values = jnp.zeros((len(obs_times), 1))

    def model():
        with Filter(filter_config=ContinuousTimeEKFConfig()):
            _continuous_lti_model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        model()

    assert "f_forecasted_state_means" in tr
    assert "f_forecasted_state_covs" in tr
    forecast_means = tr["f_forecasted_state_means"]["value"]
    forecast_covs = tr["f_forecasted_state_covs"]["value"]
    assert forecast_means.shape[0] == len(predict_times)
    _assert_all_finite(forecast_means, forecast_covs)


def test_discrete_simulator_emits_prediction_sites():
    obs_times = jnp.arange(0.0, 4.0, 1.0)
    predict_times = jnp.arange(4.0, 7.0, 1.0)

    def model():
        with DiscreteTimeSimulator():
            _discrete_lti_model(obs_times=obs_times, predict_times=predict_times)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        model()

    assert "prediction_times" in tr
    assert "predicted_states" in tr
    assert "predicted_observations" in tr
    assert tr["prediction_times"]["value"].shape[0] == len(predict_times)
    _assert_all_finite(
        tr["states"]["value"],
        tr["observations"]["value"],
        tr["predicted_states"]["value"],
        tr["predicted_observations"]["value"],
    )


def test_ode_simulator_prediction_rollout_continues_from_last_observation_state():
    obs_times = jnp.array([0.0, 0.5, 1.0])
    predict_times = jnp.array([1.2, 1.4, 1.6])

    # Deterministic scalar dynamics: dx/dt = -x
    dynamics = DynamicalModel(
        state_dim=1,
        observation_dim=1,
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.array([1.0]),
            covariance_matrix=jnp.array([[1e-8]]),
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=lambda x, u, t: -x),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[1e-8]]),
        ),
    )

    def model():
        with ODESimulator():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                predict_times=predict_times,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        model()

    states_obs = tr["states"]["value"].squeeze(-1)
    states_pred = tr["predicted_states"]["value"].squeeze(-1)
    _assert_all_finite(states_obs, states_pred, tr["predicted_observations"]["value"])
    x_last = states_obs[-1]
    expected_pred = x_last * jnp.exp(-(predict_times - obs_times[-1]))
    assert jnp.allclose(states_pred, expected_pred, atol=5e-3)


def test_sde_simulator_prediction_rollout_continues_from_last_observation_state():
    obs_times = jnp.array([0.0, 0.5, 1.0])
    predict_times = jnp.array([1.2, 1.4, 1.6])

    # Use SDESimulator with zero diffusion: deterministic dynamics in SDE pathway.
    dynamics = DynamicalModel(
        state_dim=1,
        observation_dim=1,
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.array([1.0]),
            covariance_matrix=jnp.array([[1e-8]]),
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -x,
            diffusion_coefficient=lambda x, u, t: jnp.zeros((1, 1)),
            bm_dim=1,
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[1e-8]]),
        ),
    )

    def model():
        with SDESimulator():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                predict_times=predict_times,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(1)):
        model()

    states_obs = tr["states"]["value"].squeeze(-1)
    states_pred = tr["predicted_states"]["value"].squeeze(-1)
    _assert_all_finite(states_obs, states_pred, tr["predicted_observations"]["value"])
    x_last = states_obs[-1]
    expected_pred = x_last * jnp.exp(-(predict_times - obs_times[-1]))
    assert jnp.allclose(states_pred, expected_pred, atol=5e-3)


def test_unsupported_cuthbert_predict_times_raises_useful_error():
    obs_times = jnp.arange(0.0, 5.0, 1.0)
    predict_times = jnp.arange(5.0, 8.0, 1.0)
    obs_values = jnp.zeros((len(obs_times), 1))

    def model():
        with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
            _discrete_lti_model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    with pytest.raises(ValueError, match="not supported for cuthbert"):
        with seed(rng_seed=jr.PRNGKey(0)):
            model()

