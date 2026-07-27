"""Tests for the state-path builder handler."""

from typing import cast

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from jaxtyping import Array
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx.inference.utils.distribution_utils import (
    _ForwardSimulationImproperUniform,
)
from dynestyx.observation_missingness import prepare_missing_observation_metadata


def _make_discrete_dynamics():
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(0.8 * x, 0.3),
        observation_model=lambda x, u, t: dist.Normal(x, 0.25),
    )


def _make_ode_dynamics():
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 0.7),
        state_evolution=dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: 0.0 * x),
        observation_model=lambda x, u, t: dist.Normal(x, 0.2),
    )


def _make_dirac_ode_dynamics():
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 0.7),
        state_evolution=dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: 0.0 * x),
        observation_model=dsx.DiracIdentityObservation(),
    )


def _make_dirac_discrete_dynamics():
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=0.5 * jnp.eye(2),
        ),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            loc=0.8 * x,
            covariance_matrix=0.2 * jnp.eye(2),
        ),
        observation_model=dsx.DiracIdentityObservation(),
    )


def _make_vector_gaussian_dynamics(alpha=0.8):
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=0.5 * jnp.eye(2),
        ),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            loc=alpha * x,
            covariance_matrix=0.2 * jnp.eye(2),
        ),
        observation_model=lambda x, u, t: dist.MultivariateNormal(
            loc=x,
            covariance_matrix=0.3 * jnp.eye(2),
        ),
    )


def _make_student_t_discrete_dynamics(alpha=0.8):
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=0.5 * jnp.eye(2),
        ),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            loc=alpha * x,
            covariance_matrix=0.2 * jnp.eye(2),
        ),
        observation_model=lambda x, u, t: dist.MultivariateStudentT(
            df=5.0,
            loc=x,
            scale_tril=jnp.array([[0.4, 0.0], [0.15, 0.5]]),
        ),
    )


def _make_deterministic_rollout_dynamics():
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Delta(jnp.array(0.0)),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(x + 1.0),
        observation_model=lambda x, u, t: dist.Delta(2.0 * x),
    )


def _make_deterministic_prior_dynamics():
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Delta(jnp.array(3.0)),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(x + 2.0),
        observation_model=lambda x, u, t: dist.Normal(x, 0.25),
    )


def _manual_discrete_state_log_prob(dynamics, state_path, state_path_times):
    expected = dynamics.initial_condition.log_prob(state_path[0])
    for idx in range(len(state_path_times) - 1):
        expected = expected + dynamics.state_evolution(
            state_path[idx],
            None,
            state_path_times[idx],
            state_path_times[idx + 1],
        ).log_prob(state_path[idx + 1])
    return expected


def test_latent_path_builder_sample_discrete_matches_log_prob():
    dynamics = _make_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.2, -0.1, 0.3])
    state_path_params = jnp.array([0.1, -0.2, 0.4])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

    assert jnp.array_equal(tr["f_state_path_params"]["value"], state_path_params)
    assert jnp.array_equal(tr["f_state_path"]["value"], state_path_params)
    assert jnp.array_equal(tr["f_state_path_param_times"]["value"], obs_times)
    assert jnp.array_equal(tr["f_state_path_times"]["value"], obs_times)
    assert jnp.allclose(
        tr["f_joint_log_prob"]["value"],
        dsx.log_prob(
            dynamics,
            state_path_params=state_path_params,
            state_path_param_times=obs_times,
            obs_times=obs_times,
            obs_values=obs_values,
        ),
    )


def test_latent_path_builder_chunk_size_matches_unchunked_scoring():
    dynamics = _make_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    obs_values = jnp.array([0.2, -0.1, 0.3, 0.0])
    state_path_params = jnp.array([0.1, -0.2, 0.4, 0.05])

    with trace() as chunked_tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder(chunk_size=2):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

    with trace() as unchunked_tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

    assert jnp.allclose(
        chunked_tr["f_joint_log_prob"]["value"],
        unchunked_tr["f_joint_log_prob"]["value"],
    )


def test_latent_path_builder_sample_ode_reconstructs_state_path():
    dynamics = _make_ode_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.1, 0.1, 0.1])
    state_path_params = jnp.array(0.1)
    ode_simulator_config = dsx.ODESimulatorConfig(dt0=0.25, max_steps=100)
    latent_path_builder = dsx.LatentPathBuilder(
        ode_simulator_config=ode_simulator_config
    )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with latent_path_builder:
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

    assert latent_path_builder.ode_simulator_config is ode_simulator_config
    assert jnp.array_equal(tr["f_state_path_param_times"]["value"], jnp.array([0.0]))
    assert jnp.array_equal(
        tr["f_state_path_times"]["value"], jnp.array([0.0, 0.0, 1.0, 2.0])
    )
    assert jnp.allclose(tr["f_state_path"]["value"], jnp.array([0.1, 0.1, 0.1, 0.1]))


def test_latent_path_builder_rejects_dirac_ode_inference():
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.1, 0.1, 0.1])

    with pytest.raises(
        ValueError,
        match="Inference/scoring .* DiracIdentityObservation",
    ):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                _make_dirac_ode_dynamics(),
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=jnp.array(0.1),
            )


def test_latent_path_builder_sample_dirac_partial_missing_compresses_per_coordinate():
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [jnp.nan, jnp.nan],
        ]
    )
    state_path_params = jnp.array([0.5, -0.3, 0.7, 0.9])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

    expected_states = jnp.array(
        [
            [0.2, 0.5],
            [-0.3, -0.1],
            [0.7, 0.9],
        ]
    )
    assert jnp.array_equal(tr["f_state_path_params"]["value"], state_path_params)
    assert jnp.array_equal(
        tr["f_state_path_param_times"]["value"], jnp.array([0.0, 1.0, 2.0, 2.0])
    )
    assert jnp.array_equal(
        tr["f_state_path_param_coordinate_indices"]["value"],
        jnp.array([1, 0, 0, 1], dtype=jnp.int32),
    )
    assert jnp.allclose(tr["f_state_path"]["value"], expected_states)
    assert jnp.allclose(
        tr["f_joint_log_prob"]["value"],
        _manual_discrete_state_log_prob(dynamics, expected_states, obs_times),
    )


def test_latent_path_builder_future_only_rollout_forwards_to_outer_simulator():
    dynamics = _make_deterministic_rollout_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.0, 2.0, 4.0])
    state_path_params = jnp.array([0.0, 1.0, 2.0])
    predict_times = jnp.array([obs_times[-1], 3.0, 4.0])

    def conditioned_model():
        with dsx.DiscreteTimeSimulator():
            with dsx.LatentPathBuilder():
                dsx.sample(
                    "f",
                    dynamics,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    predict_times=predict_times,
                    state_path_params=state_path_params,
                )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        conditioned_model()

    assert jnp.array_equal(tr["f_state_path"]["value"], state_path_params)
    assert jnp.array_equal(tr["f_predicted_times"]["value"], predict_times[None, :])
    assert jnp.array_equal(
        tr["f_predicted_states"]["value"][0, :, 0],
        jnp.array([2.0, 3.0, 4.0]),
    )
    assert jnp.array_equal(
        tr["f_predicted_observations"]["value"][0, :, 0],
        jnp.array([4.0, 6.0, 8.0]),
    )


def test_latent_path_builder_rejects_in_window_predict_times():
    dynamics = _make_deterministic_rollout_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.0, 2.0, 4.0])

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="LatentPathBuilder rollout only supports predict_times >= max\\(state_path_times\\)",
    ):
        with dsx.DiscreteTimeSimulator():
            with dsx.LatentPathBuilder():
                dsx.sample(
                    "f",
                    dynamics,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    predict_times=jnp.array([1.0, 2.0, 3.0]),
                    state_path_params=jnp.array([0.0, 1.0, 2.0]),
                )


def test_latent_path_builder_future_only_rollout_ode_states():
    dynamics = _make_ode_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.1, 0.1, 0.1])
    predict_times = jnp.array([obs_times[-1], 3.0, 4.0])

    def conditioned_model():
        with dsx.ODESimulator():
            with dsx.LatentPathBuilder():
                dsx.sample(
                    "f",
                    dynamics,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    predict_times=predict_times,
                    state_path_params=jnp.array(0.1),
                )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        conditioned_model()

    assert jnp.array_equal(tr["f_predicted_times"]["value"], predict_times[None, :])
    assert jnp.allclose(
        jnp.squeeze(tr["f_predicted_states"]["value"][0]),
        jnp.array([0.1, 0.1, 0.1]),
    )


def test_prepare_missing_observation_metadata_matches_dirac_partial_missing_layout():
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [jnp.nan, jnp.nan],
        ]
    )

    metadata = prepare_missing_observation_metadata(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )

    assert jnp.array_equal(
        metadata.missing_obs_times,
        jnp.array([0.0, 1.0, 2.0, 2.0]),
    )
    assert metadata.missing_obs_coordinate_indices is not None
    assert jnp.array_equal(
        metadata.missing_obs_coordinate_indices,
        jnp.array([1, 0, 0, 1], dtype=jnp.int32),
    )
    assert jnp.array_equal(metadata.missing_flat_indices, jnp.array([1, 2, 4, 5]))
    assert metadata.observation_shape == (3, 2)


def test_latent_path_builder_rejects_dsx_condition():
    dynamics = _make_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0])
    obs_values = jnp.array([0.2, -0.1])

    with pytest.raises(ValueError, match="only supports dsx.sample"):
        with dsx.LatentPathBuilder():
            dsx.condition(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=jnp.array([0.1, -0.2]),
            )


def test_latent_path_builder_sample_registers_expected_sites():
    dynamics = _make_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.2, -0.1, 0.3])
    state_path_params = jnp.array([0.1, -0.2, 0.4])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

    assert "f_state_path_params" in tr
    assert "f_joint_log_prob_factor" in tr
    assert isinstance(tr["f_state_path_params"]["fn"], dist.ImproperUniform)
    assert tr["f_state_path_params"]["fn"].log_prob(state_path_params) == 0.0
    assert "f_state_path_params_base_log_prob_correction" not in tr
    assert "f_state_path_param_times" in tr
    assert "f_state_path" in tr
    assert "f_state_path_times" in tr
    assert "f_joint_log_prob" in tr
    assert jnp.array_equal(tr["f_state_path_params"]["value"], state_path_params)
    assert jnp.array_equal(tr["f_state_path"]["value"], state_path_params)

    expected = dsx.log_prob(
        dynamics,
        state_path_params=state_path_params,
        state_path_param_times=obs_times,
        obs_times=obs_times,
        obs_values=obs_values,
    )
    actual = tr["f_joint_log_prob_factor"]["fn"].log_factor
    assert jnp.allclose(actual, expected)


def test_latent_path_builder_sample_registers_dirac_index_metadata():
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [jnp.nan, jnp.nan],
        ]
    )
    state_path_params = jnp.array([0.5, -0.3, 0.7, 0.9])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

    expected_states = jnp.array(
        [
            [0.2, 0.5],
            [-0.3, -0.1],
            [0.7, 0.9],
        ]
    )
    assert "f_state_path_param_coordinate_indices" in tr
    assert jnp.array_equal(tr["f_state_path_params"]["value"], state_path_params)
    assert jnp.array_equal(
        tr["f_state_path_param_coordinate_indices"]["value"],
        jnp.array([1, 0, 0, 1], dtype=jnp.int32),
    )
    assert jnp.allclose(tr["f_state_path"]["value"], expected_states)

    expected = _manual_discrete_state_log_prob(dynamics, expected_states, obs_times)
    actual = tr["f_joint_log_prob_factor"]["fn"].log_factor
    assert jnp.allclose(actual, expected)


def test_latent_path_builder_sample_can_draw_latents_when_unspecified():
    dynamics = _make_deterministic_prior_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([3.0, 5.0, 7.0])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    assert "f_state_path_params" in tr
    assert "f_state_path" in tr
    assert jnp.array_equal(
        tr["f_state_path_params"]["value"],
        jnp.array([3.0, 5.0, 7.0]),
    )


def test_forward_simulation_improper_uniform_rsample_and_sample_shape():
    prior = _ForwardSimulationImproperUniform(
        eqx.Partial(jr.normal, shape=(2,)),
        event_shape=(2,),
    )
    key = jr.PRNGKey(12)

    samples = prior.sample(key, sample_shape=(2, 3))
    resamples = prior.rsample(key, sample_shape=(2, 3))

    assert samples.shape == (2, 3, 2)
    assert jnp.array_equal(samples, resamples)
    assert jnp.array_equal(prior.log_prob(samples), jnp.zeros((2, 3)))


def test_latent_path_builder_bare_predictive_draws_dynamical_prior_paths():
    dynamics = _make_deterministic_prior_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([3.0, 5.0, 7.0])

    def model():
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    predictive = Predictive(model, num_samples=3)(jr.PRNGKey(13))

    expected = jnp.broadcast_to(jnp.array([3.0, 5.0, 7.0]), (3, 3))
    assert jnp.array_equal(predictive["f_state_path_params"], expected)
    assert jnp.array_equal(predictive["f_state_path"], expected)


def test_latent_path_builder_ode_prior_site_samples_initial_condition():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Delta(jnp.array(2.5)),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.zeros_like(x)
        ),
        observation_model=lambda x, u, t: dist.Normal(x, 0.2),
    )
    obs_times = jnp.array([0.0, 1.0, 2.0])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(14)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=jnp.full((3,), 2.5),
            )

    assert tr["f_state_path_params"]["value"].shape == (1,)
    assert jnp.array_equal(tr["f_state_path_params"]["value"], jnp.array([2.5]))
    assert jnp.allclose(tr["f_state_path"]["value"], 2.5)


def test_latent_path_builder_dirac_prior_projects_only_free_coordinates():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Delta(jnp.array([0.0, 1.0]), event_dim=1),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(
            x + jnp.array([10.0, 20.0]),
            event_dim=1,
        ),
        observation_model=dsx.DiracIdentityObservation(),
    )
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([[0.0, jnp.nan], [jnp.nan, 21.0], [jnp.nan, jnp.nan]])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(15)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    assert jnp.array_equal(
        tr["f_state_path_params"]["value"],
        jnp.array([1.0, 10.0, 20.0, 41.0]),
    )
    assert jnp.array_equal(
        tr["f_state_path"]["value"],
        jnp.array([[0.0, 1.0], [10.0, 21.0], [20.0, 41.0]]),
    )


def test_latent_path_builder_dirac_fully_observed_has_zero_free_latents():
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0])
    obs_values = jnp.array([[0.1, -0.2], [0.3, 0.4]])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    assert tr["f_state_path_params"]["value"].shape == (0,)
    assert jnp.array_equal(
        tr["f_state_path_param_coordinate_indices"]["value"],
        jnp.array([], dtype=jnp.int32),
    )
    assert jnp.allclose(tr["f_state_path"]["value"], obs_values)


def test_latent_path_builder_dirac_partial_missing_mcmc_smoke():
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [jnp.nan, jnp.nan],
        ]
    )

    def conditioned_model(obs_times=None, obs_values=None):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    mcmc = MCMC(
        NUTS(conditioned_model),
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(0), obs_times=obs_times, obs_values=obs_values)

    samples = mcmc.get_samples()
    assert samples["f_state_path_params"].shape == (10, 4)

    posterior = Predictive(
        conditioned_model,
        posterior_samples=samples,
        return_sites=[
            "f_state_path",
            "f_state_path_param_times",
            "f_state_path_param_coordinate_indices",
        ],
        exclude_deterministic=False,
    )(jr.PRNGKey(1), obs_times=obs_times, obs_values=obs_values)

    assert jnp.array_equal(
        posterior["f_state_path_param_times"][0],
        jnp.array([0.0, 1.0, 2.0, 2.0]),
    )
    assert jnp.array_equal(
        posterior["f_state_path_param_coordinate_indices"][0],
        jnp.array([1, 0, 0, 1], dtype=jnp.int32),
    )
    assert posterior["f_state_path"].shape == (10, 3, 2)


def test_latent_path_builder_dirac_partial_missing_predictive_keeps_compressed_layout():
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [jnp.nan, jnp.nan],
        ]
    )

    def conditioned_model(obs_times=None, obs_values=None):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        conditioned_model(obs_times=obs_times, obs_values=obs_values)

    assert tr["f_state_path_params"]["value"].shape == (4,)
    assert jnp.array_equal(
        tr["f_state_path_param_coordinate_indices"]["value"],
        jnp.array([1, 0, 0, 1], dtype=jnp.int32),
    )

    predictive = Predictive(conditioned_model, num_samples=2)(
        jr.PRNGKey(1),
        obs_times=obs_times,
        obs_values=obs_values,
    )

    assert predictive["f_state_path_params"].shape == (2, 4)
    assert jnp.array_equal(
        predictive["f_state_path_param_coordinate_indices"][0],
        jnp.array([1, 0, 0, 1], dtype=jnp.int32),
    )


def test_latent_path_builder_dirac_partial_missing_explicit_augment_uses_state_path_params():
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [jnp.nan, jnp.nan],
        ]
    )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder(missing_observation_strategy="augment"):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    assert tr["f_state_path_params"]["value"].shape == (4,)
    assert "f_missing_obs_values" not in tr
    assert jnp.array_equal(
        tr["f_state_path_param_coordinate_indices"]["value"],
        jnp.array([1, 0, 0, 1], dtype=jnp.int32),
    )


@pytest.mark.parametrize("strategy", ["marginalize", "error"])
def test_latent_path_builder_dirac_partial_missing_rejects_non_augment_strategies(
    strategy,
):
    dynamics = _make_dirac_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [jnp.nan, jnp.nan],
        ]
    )

    with pytest.raises(
        ValueError,
        match="supports only augment semantics",
    ):
        with dsx.LatentPathBuilder(missing_observation_strategy=strategy):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )


def test_latent_path_builder_sample_registers_missing_observation_sites_under_augment():
    dynamics = _make_student_t_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array(
        [
            [0.2, jnp.nan],
            [jnp.nan, -0.1],
            [0.3, 0.4],
        ]
    )
    state_path_params = jnp.array(
        [
            [0.1, -0.2],
            [0.2, -0.3],
            [0.3, -0.1],
        ]
    )
    metadata = prepare_missing_observation_metadata(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )
    missing_obs_values = jnp.array([0.5, -0.4])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder(missing_observation_strategy="augment"):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
                missing_obs_values=missing_obs_values,
            )

    completed_obs = jnp.array(
        [
            [0.2, 0.5],
            [-0.4, -0.1],
            [0.3, 0.4],
        ]
    )
    expected = dsx.log_prob(
        dynamics,
        state_path_params=state_path_params,
        state_path_param_times=obs_times,
        obs_times=obs_times,
        obs_values=obs_values,
        missing_observation_strategy="augment",
        missing_obs_values=missing_obs_values,
        missing_obs_metadata=metadata,
    )

    assert "f_missing_obs_values" in tr
    assert isinstance(tr["f_state_path_params"]["fn"], dist.ImproperUniform)
    assert isinstance(tr["f_missing_obs_values"]["fn"], dist.ImproperUniform)
    assert tr["f_state_path_params"]["fn"].log_prob(state_path_params) == 0.0
    assert tr["f_missing_obs_values"]["fn"].log_prob(missing_obs_values) == 0.0
    assert "f_state_path_params_base_log_prob_correction" not in tr
    assert "f_missing_obs_base_log_prob_correction" not in tr
    assert "f_missing_obs_times" in tr
    assert "f_missing_obs_coordinate_indices" in tr
    assert "f_completed_obs_values" in tr
    assert jnp.array_equal(tr["f_missing_obs_values"]["value"], missing_obs_values)
    assert jnp.array_equal(
        tr["f_missing_obs_times"]["value"],
        jnp.array([0.0, 1.0]),
    )
    assert jnp.array_equal(
        tr["f_missing_obs_coordinate_indices"]["value"],
        jnp.array([1, 0], dtype=jnp.int32),
    )
    assert jnp.allclose(tr["f_completed_obs_values"]["value"], completed_obs)
    actual = tr["f_joint_log_prob_factor"]["fn"].log_factor
    assert jnp.allclose(actual, expected)


def test_latent_path_builder_forced_augment_on_gaussian_partial_missing_creates_sites():
    dynamics = _make_vector_gaussian_dynamics()
    obs_times = jnp.array([0.0, 1.0])
    obs_values = jnp.array([[0.2, jnp.nan], [jnp.nan, -0.1]])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder(missing_observation_strategy="augment"):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    assert "f_state_path_params" in tr
    assert "f_missing_obs_values" in tr
    assert "f_completed_obs_values" in tr
    assert tr["f_missing_obs_values"]["value"].shape == (2,)


def test_latent_path_builder_augmented_site_samples_conditional_observation_prior():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Delta(jnp.array(0.0)),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(x + 1.0),
        observation_model=lambda x, u, t: dist.Delta(
            jnp.stack([jnp.ravel(x)[0] + 5.0, jnp.ravel(x)[0] + 7.0]),
            event_dim=1,
        ),
    )
    obs_times = jnp.array([0.0, 1.0])
    obs_values = jnp.array([[5.0, jnp.nan], [jnp.nan, 8.0]])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(16)):
        with dsx.LatentPathBuilder(missing_observation_strategy="augment"):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    assert jnp.array_equal(tr["f_state_path"]["value"], jnp.array([0.0, 1.0]))
    assert jnp.array_equal(
        tr["f_missing_obs_values"]["value"],
        jnp.array([7.0, 6.0]),
    )
    assert jnp.array_equal(
        tr["f_completed_obs_values"]["value"],
        jnp.array([[5.0, 7.0], [6.0, 8.0]]),
    )


def test_latent_path_builder_auto_augments_student_t_partial_missing_mcmc_smoke():
    obs_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    true_alpha = 0.3
    dynamics_true = _make_student_t_discrete_dynamics(alpha=true_alpha)
    sim = dsx.simulate(
        dynamics_true,
        rng_key=jr.PRNGKey(2),
        predict_times=obs_times,
    )
    obs_values = cast(Array, sim.observations)[0]
    obs_values = obs_values.at[1, 0].set(jnp.nan)
    obs_values = obs_values.at[2, 1].set(jnp.nan)

    def conditioned_model(obs_times=None, obs_values=None):
        alpha = numpyro.sample("alpha", dist.Uniform(0.0, 0.9))
        dynamics = _make_student_t_discrete_dynamics(alpha=alpha)
        with dsx.LatentPathBuilder(missing_observation_strategy="auto"):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )

    mcmc = MCMC(
        NUTS(conditioned_model),
        num_warmup=10,
        num_samples=10,
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(3), obs_times=obs_times, obs_values=obs_values)

    samples = mcmc.get_samples()
    assert "alpha" in samples
    assert samples["f_missing_obs_values"].shape == (10, 2)

    posterior = Predictive(
        conditioned_model,
        posterior_samples=samples,
        return_sites=[
            "f_completed_obs_values",
            "f_missing_obs_times",
            "f_missing_obs_coordinate_indices",
        ],
        exclude_deterministic=False,
    )(jr.PRNGKey(4), obs_times=obs_times, obs_values=obs_values)

    assert posterior["f_completed_obs_values"].shape == (10, 4, 2)
    assert jnp.array_equal(
        posterior["f_missing_obs_coordinate_indices"][0],
        jnp.array([0, 1], dtype=jnp.int32),
    )


def test_latent_path_builder_rejects_native_sdes():
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.1 * x,
            diffusion=dsx.ScalarDiffusion(jnp.array(0.2), bm_dim=1),
        ),
        observation_model=lambda x, u, t: dist.Normal(x, 1.0),
    )

    with pytest.raises(ValueError, match="discretize"):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=jnp.array([0.0]),
                obs_values=jnp.array([0.1]),
                state_path_params=jnp.array([0.2]),
            )
