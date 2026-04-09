import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, ODESimulator, Smoother
from dynestyx.inference.filter_configs import (
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    EnKFConfig,
    HMMConfig,
    KFConfig,
    PFConfig,
)
from tests.models import (
    discrete_time_lti_simplified_model,
    hmm_model,
    jumpy_controls_model_ode,
)


def _gen_obs_discrete():
    rng = jr.PRNGKey(42)
    obs_times = jnp.arange(0.0, 6.0, 1.0)
    predict_times = jnp.arange(0.0, 9.0, 1.0)
    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            discrete_time_lti_simplified_model,
            params={"alpha": jnp.array(0.35)},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(rng, predict_times=obs_times)
    obs_values = sim["f_observations"][0, 0]
    return obs_times, obs_values, predict_times


def _gen_obs_ode():
    rng = jr.PRNGKey(42)
    obs_times = jnp.linspace(0.0, 0.5, 11)
    predict_times = jnp.linspace(0.0, 1.0, 21)
    ctrl_times = predict_times
    ctrl_values = jnp.ones((len(predict_times),)) * 100
    for i in range(1, len(ctrl_values), 2):
        ctrl_values = ctrl_values.at[i].set(-ctrl_values[i])
    with ODESimulator(n_simulations=1):
        pred = Predictive(
            jumpy_controls_model_ode,
            params={},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(
            rng,
            predict_times=obs_times,
            ctrl_times=obs_times,
            ctrl_values=ctrl_values[: len(obs_times)],
        )
    obs_values = sim["f_observations"][0, 0]
    return obs_times, obs_values, predict_times, ctrl_times, ctrl_values


def test_predictive_smoother_discretetimesimulator_shapes():
    obs_times, obs_values, predict_times = _gen_obs_discrete()

    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params={"alpha": jnp.array(0.35)},
        num_samples=2,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator(n_simulations=2):
        with Smoother(
            filter_config=KFConfig(
                filter_source="cd_dynamax",
                record_smoothed_states_mean=True,
                record_smoothed_states_cov_diag=True,
            )
        ):
            samples = predictive(
                jr.PRNGKey(0),
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    assert samples["f_predicted_states"].shape == (2, 2, len(predict_times), 2)
    assert samples["f_predicted_times"].shape == (2, 2, len(predict_times))
    assert samples["f_smoothed_states_mean"].shape == (2, len(obs_times), 2)
    assert samples["f_smoothed_states_cov_diag"].shape == (2, len(obs_times), 2)


def test_predictive_smoother_odesimulator_shapes():
    obs_times, obs_values, predict_times, ctrl_times, ctrl_values = _gen_obs_ode()

    predictive = Predictive(
        jumpy_controls_model_ode,
        params={},
        num_samples=2,
        exclude_deterministic=False,
    )
    with ODESimulator(n_simulations=2):
        with Smoother(
            filter_config=ContinuousTimeEKFConfig(record_smoothed_states_mean=True)
        ):
            samples = predictive(
                jr.PRNGKey(0),
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

    assert samples["f_predicted_states"].shape == (2, 2, len(predict_times), 1)
    assert samples["f_predicted_times"].shape == (2, 2, len(predict_times))
    assert samples["f_smoothed_states_mean"].shape == (2, len(obs_times), 1)


def test_predictive_smoother_cuthbert_pf_records_particles():
    obs_times, obs_values, predict_times = _gen_obs_discrete()

    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params={"alpha": jnp.array(0.35)},
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator(n_simulations=1):
        with Smoother(
            filter_config=PFConfig(
                filter_source="cuthbert",
                n_particles=32,
                record_smoothed_particles=True,
                record_smoothed_log_weights=True,
                record_smoothed_states_mean=True,
            )
        ):
            samples = predictive(
                jr.PRNGKey(0),
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    assert samples["f_smoothed_particles"].shape[0] == 1
    assert samples["f_smoothed_particles"].shape[1] == len(obs_times)
    assert samples["f_smoothed_log_weights"].shape[0] == 1
    assert samples["f_smoothed_log_weights"].shape[1] == len(obs_times)
    assert jnp.all(jnp.isfinite(samples["f_smoothed_log_weights"]))


def test_smoother_unsupported_discrete_enkf_errors():
    obs_times, obs_values, _ = _gen_obs_discrete()

    with pytest.raises(ValueError, match="EnKF smoothing is not supported"):
        with seed(rng_seed=jr.PRNGKey(0)):
            with Smoother(filter_config=EnKFConfig(filter_source="cd_dynamax")):
                discrete_time_lti_simplified_model(
                    obs_times=obs_times,
                    obs_values=obs_values,
                )


def test_smoother_unsupported_continuous_enkf_errors():
    obs_times, obs_values, _, _, _ = _gen_obs_ode()

    with pytest.raises(ValueError, match="Supported continuous-time smoothers"):
        with seed(rng_seed=jr.PRNGKey(0)):
            with Smoother(filter_config=ContinuousTimeEnKFConfig(n_particles=8)):
                jumpy_controls_model_ode(
                    obs_times=obs_times,
                    obs_values=obs_values,
                )


def test_smoother_unsupported_hmm_errors():
    obs_times = jnp.arange(0.0, 4.0, 1.0)
    params = {
        "A": jnp.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]]),
        "mu": jnp.array([-1.0, 0.0, 1.0]),
        "sigma": jnp.array(0.5),
    }

    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            hmm_model,
            params=params,
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(
            jr.PRNGKey(1),
            predict_times=obs_times,
            ctrl_times=obs_times,
            ctrl_values=jnp.zeros((len(obs_times), 1)),
        )

    obs_values = sim["f_observations"][0, 0]
    with pytest.raises(ValueError, match="HMM smoothing is not supported"):
        with seed(rng_seed=jr.PRNGKey(0)):
            with Smoother(filter_config=HMMConfig()):
                hmm_model(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=obs_times,
                    ctrl_values=jnp.zeros((len(obs_times), 1)),
                )


def _plate_discrete_model(obs_times=None, obs_values=None, predict_times=None, m=2):
    dynamics = dsx.LTI_discrete(
        A=jnp.array([[0.9, 0.0], [0.0, 0.8]]),
        Q=0.1 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.25]]),
    )
    with dsx.plate("trajectories", m):
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def test_smoother_plate_batched_loglik_shape():
    obs_times = jnp.arange(0.0, 5.0, 1.0)
    m = 3
    obs_values = jnp.zeros((m, len(obs_times), 1))

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with Smoother(filter_config=KFConfig(filter_source="cd_dynamax")):
            _plate_discrete_model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=obs_times,
                m=m,
            )

    assert tr["f_marginal_loglik"]["value"].shape == (m,)


def _explicit_rollout_metadata_model(predict_times=None):
    dynamics = dsx.LTI_discrete(
        A=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
        Q=0.01 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.1]]),
    )
    filtered_dists = [
        dist.MultivariateNormal(jnp.zeros(2), covariance_matrix=jnp.eye(2))
    ]
    dsx.sample(
        "f",
        dynamics,
        predict_times=predict_times,
        filtered_times=jnp.array([0.0]),
        filtered_dists=filtered_dists,
        smoothed_times=jnp.array([0.0]),
        smoothed_dists=None,
    )


def test_simulator_prioritizes_smoothed_rollout_metadata_when_both_present():
    with pytest.raises(ValueError, match="smoothed_times"):
        with seed(rng_seed=jr.PRNGKey(0)):
            with DiscreteTimeSimulator(n_simulations=1):
                _explicit_rollout_metadata_model(
                    predict_times=jnp.arange(0.0, 3.0, 1.0)
                )
