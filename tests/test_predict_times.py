import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx.inference.filter_configs import (
    ContinuousTimeEKFConfig,
    EKFConfig,
    KFConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
    LinearGaussianObservation,
)
from dynestyx.models.lti_dynamics import LTI_continuous, LTI_discrete
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


def test_discrete_simulator_allows_predict_times_without_obs_times():
    predict_times = jnp.array([1.0, 2.0, 3.0])

    def model():
        with DiscreteTimeSimulator():
            _discrete_lti_model(predict_times=predict_times)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(11)):
        model()

    assert "prediction_times" in tr
    assert "predicted_states" in tr
    assert "predicted_observations" in tr
    assert tr["prediction_times"]["value"].shape[0] == len(predict_times)
    _assert_all_finite(
        tr["predicted_states"]["value"],
        tr["predicted_observations"]["value"],
    )


def test_discrete_dirac_observation_allows_forecast_with_obs_conditioning():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    predict_times = jnp.array([3.0, 4.0])
    obs_values = jnp.array([[0.2], [0.1], [-0.3]])

    dynamics = DynamicalModel(
        state_dim=1,
        observation_dim=1,
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(0.8 * x, 0.2),
        observation_model=DiracIdentityObservation(),
    )

    def model():
        with DiscreteTimeSimulator():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(12)):
        model()

    assert "states" in tr
    assert "observations" in tr
    assert "prediction_times" in tr
    assert "predicted_states" in tr
    assert "predicted_observations" in tr
    assert tr["prediction_times"]["value"].shape[0] == len(predict_times)
    assert jnp.allclose(tr["states"]["value"], obs_values)
    assert jnp.allclose(tr["observations"]["value"], obs_values)
    _assert_all_finite(
        tr["predicted_states"]["value"],
        tr["predicted_observations"]["value"],
    )


def test_discrete_dirac_predictive_predict_only_rollout_uses_t0_anchor():
    predict_times = jnp.array([1.0, 2.0, 3.0])

    dynamics = DynamicalModel(
        state_dim=1,
        observation_dim=1,
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(0.8 * x, event_dim=0),
        observation_model=DiracIdentityObservation(),
        t0=0.0,
    )

    def model_predict_only(predict_times=None):
        with DiscreteTimeSimulator():
            dsx.sample("f", dynamics, predict_times=predict_times)

    x0_samples = jnp.array([[1.0], [2.0], [-1.5]])
    out = Predictive(
        model_predict_only,
        posterior_samples={"x_0": x0_samples},
        exclude_deterministic=False,
    )(jr.PRNGKey(55), predict_times=predict_times)

    pred_states = out["predicted_states"]
    pred_obs = out["predicted_observations"]
    assert pred_states.shape == (len(x0_samples), len(predict_times), 1)
    assert pred_obs.shape == (len(x0_samples), len(predict_times), 1)
    _assert_all_finite(pred_states, pred_obs)

    expected = jnp.stack(
        [x0_samples[:, 0] * (0.8 ** (k + 1)) for k in range(len(predict_times))],
        axis=1,
    )[:, :, None]
    assert jnp.allclose(pred_states, expected)
    assert jnp.allclose(pred_obs, expected)
    # Regression check: first predicted step should not simply replay x_0.
    assert not jnp.allclose(pred_states[:, 0, :], x0_samples)


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


def test_ode_simulator_allows_predict_times_without_obs_times():
    predict_times = jnp.array([1.0, 1.2, 1.4])

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
            dsx.sample("f", dynamics, predict_times=predict_times)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(2)):
        model()

    assert "prediction_times" in tr
    assert "predicted_states" in tr
    assert "predicted_observations" in tr
    x0 = tr["x_0"]["value"].squeeze(-1)
    states_pred = tr["predicted_states"]["value"].squeeze(-1)
    expected_pred = x0 * jnp.exp(-predict_times)
    _assert_all_finite(states_pred, tr["predicted_observations"]["value"])
    assert jnp.allclose(states_pred, expected_pred, atol=5e-3)


def test_sde_simulator_allows_predict_times_without_obs_times():
    predict_times = jnp.array([1.0, 1.2, 1.4])

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
            dsx.sample("f", dynamics, predict_times=predict_times)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(3)):
        model()

    assert "prediction_times" in tr
    assert "predicted_states" in tr
    assert "predicted_observations" in tr
    x0 = tr["x_0"]["value"].squeeze(-1)
    states_pred = tr["predicted_states"]["value"].squeeze(-1)
    expected_pred = x0 * jnp.exp(-predict_times)
    _assert_all_finite(states_pred, tr["predicted_observations"]["value"])
    assert jnp.allclose(states_pred, expected_pred, atol=5e-3)


def test_ode_simulator_predict_times_without_obs_times_respects_nonzero_t0():
    predict_times = jnp.array([1.0, 1.2, 1.4])
    t0 = 0.5

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
        t0=t0,
    )

    def model():
        with ODESimulator():
            dsx.sample("f", dynamics, predict_times=predict_times)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(4)):
        model()

    x0 = tr["x_0"]["value"].squeeze(-1)
    states_pred = tr["predicted_states"]["value"].squeeze(-1)
    expected_pred = x0 * jnp.exp(-(predict_times - t0))
    assert jnp.allclose(states_pred, expected_pred, atol=5e-3)


def test_ode_long_horizon_predict_only_matches_rollout_with_earlier_obs_times():
    t0 = 0.0
    obs_times = jnp.arange(0.1, 5.0, 0.1)
    predict_times = jnp.arange(5.0, 20.0, 0.1)

    # Use a non-attracting system so this test validates time propagation
    # rather than trivially passing due to convergence to a fixed point.
    def _harmonic_drift(x, u, t):
        return jnp.array([x[1], -x[0]])

    dynamics = DynamicalModel(
        state_dim=2,
        observation_dim=2,
        control_dim=0,
        initial_condition=dist.Delta(jnp.array([1.0, 0.0]), event_dim=1),
        state_evolution=ContinuousTimeStateEvolution(drift=_harmonic_drift),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(2),
            R=1e-8 * jnp.eye(2),
        ),
        t0=t0,
    )
    # Condition on values from the exact solution so the observation pathway
    # is deterministic and does not alter the latent ODE trajectory.
    obs_values = jnp.stack([jnp.cos(obs_times), -jnp.sin(obs_times)], axis=-1)

    def model_with_obs():
        with ODESimulator():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    def model_predict_only():
        with ODESimulator():
            dsx.sample("f", dynamics, predict_times=predict_times)

    with trace() as tr_with_obs, seed(rng_seed=jr.PRNGKey(41)):
        model_with_obs()
    with trace() as tr_predict_only, seed(rng_seed=jr.PRNGKey(42)):
        model_predict_only()

    pred_with_obs = tr_with_obs["predicted_states"]["value"]
    pred_only = tr_predict_only["predicted_states"]["value"]
    expected = jnp.stack([jnp.cos(predict_times), -jnp.sin(predict_times)], axis=-1)

    _assert_all_finite(pred_with_obs, pred_only)
    assert pred_with_obs.shape == (len(predict_times), 2)
    assert pred_only.shape == (len(predict_times), 2)
    assert jnp.allclose(pred_with_obs, expected, atol=5e-3)
    assert jnp.allclose(pred_only, expected, atol=5e-3)
    assert jnp.allclose(pred_only, pred_with_obs, atol=5e-3)


def test_ode_predictive_predict_only_harmonic_rollout_respects_t0_integration():
    predict_times = jnp.arange(5.0, 8.0, 0.1)

    def _harmonic_drift(x, u, t):
        return jnp.array([x[1], -x[0]])

    dynamics = DynamicalModel(
        state_dim=2,
        observation_dim=2,
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=jnp.eye(2),
        ),
        state_evolution=ContinuousTimeStateEvolution(drift=_harmonic_drift),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(2),
            R=1e-8 * jnp.eye(2),
        ),
        t0=0.0,
    )

    def model_predict_only(predict_times=None):
        with ODESimulator():
            dsx.sample("f", dynamics, predict_times=predict_times)

    x0_samples = jnp.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.5],
        ]
    )
    posterior_samples = {"x_0": x0_samples}

    out = Predictive(
        model_predict_only,
        posterior_samples=posterior_samples,
        exclude_deterministic=False,
    )(jr.PRNGKey(43), predict_times=predict_times)

    pred_states = out["predicted_states"]
    assert pred_states.shape == (len(x0_samples), len(predict_times), 2)
    _assert_all_finite(pred_states)

    t = predict_times[None, :]
    x10 = x0_samples[:, 0:1]
    x20 = x0_samples[:, 1:2]
    expected = jnp.stack(
        [
            x10 * jnp.cos(t) + x20 * jnp.sin(t),
            -x10 * jnp.sin(t) + x20 * jnp.cos(t),
        ],
        axis=-1,
    )
    assert jnp.allclose(pred_states, expected, atol=5e-3)
    # Guard against a rollout bug where prediction starts from x_0 at the
    # first predict time rather than integrating from t0.
    assert not jnp.allclose(pred_states[:, 0, :], x0_samples, atol=1e-3)


def test_simulator_allows_obs_times_after_t0():
    obs_times = jnp.array([0.1, 0.2, 0.3])
    predict_times = jnp.array([0.4, 0.5])

    def model():
        with ODESimulator():
            dsx.sample(
                "f",
                DynamicalModel(
                    state_dim=1,
                    observation_dim=1,
                    control_dim=0,
                    initial_condition=dist.MultivariateNormal(
                        loc=jnp.array([1.0]),
                        covariance_matrix=jnp.array([[1e-8]]),
                    ),
                    state_evolution=ContinuousTimeStateEvolution(
                        drift=lambda x, u, t: -x
                    ),
                    observation_model=LinearGaussianObservation(
                        H=jnp.array([[1.0]]),
                        R=jnp.array([[1e-8]]),
                    ),
                    t0=0.0,
                ),
                obs_times=obs_times,
                predict_times=predict_times,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(5)):
        model()

    x0 = tr["x_0"]["value"].squeeze(-1)
    states_obs = tr["states"]["value"].squeeze(-1)
    states_pred = tr["predicted_states"]["value"].squeeze(-1)
    expected_obs = x0 * jnp.exp(-obs_times)
    expected_pred = x0 * jnp.exp(-predict_times)
    assert jnp.allclose(states_obs, expected_obs, atol=5e-3)
    assert jnp.allclose(states_pred, expected_pred, atol=5e-3)


def test_filter_raises_when_obs_times_do_not_match_t0():
    obs_times = jnp.array([1.0, 2.0, 3.0])
    obs_values = jnp.zeros((len(obs_times), 1))

    def model():
        with Filter(filter_config=KFConfig()):
            _discrete_lti_model(obs_times=obs_times, obs_values=obs_values)

    with pytest.raises(ValueError, match="t0 must equal obs_times\\[0\\]"):
        with seed(rng_seed=jr.PRNGKey(6)):
            model()


def test_simulator_raises_when_obs_times_start_before_t0():
    obs_times = jnp.array([0.0, 0.1, 0.2])

    def model():
        with ODESimulator():
            dsx.sample(
                "f",
                DynamicalModel(
                    state_dim=1,
                    observation_dim=1,
                    control_dim=0,
                    initial_condition=dist.MultivariateNormal(
                        loc=jnp.array([1.0]),
                        covariance_matrix=jnp.array([[1e-8]]),
                    ),
                    state_evolution=ContinuousTimeStateEvolution(
                        drift=lambda x, u, t: -x
                    ),
                    observation_model=LinearGaussianObservation(
                        H=jnp.array([[1.0]]),
                        R=jnp.array([[1e-8]]),
                    ),
                    t0=0.1,
                ),
                obs_times=obs_times,
            )

    with pytest.raises(
        ValueError, match="obs_times\\[0\\] must be greater than or equal"
    ):
        with seed(rng_seed=jr.PRNGKey(7)):
            model()


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
