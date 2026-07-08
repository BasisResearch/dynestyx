"""Tests for the state-path builder handler."""

from typing import cast

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from jaxtyping import Array
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx


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


def test_latent_path_builder_sample_ode_reconstructs_state_path():
    dynamics = _make_ode_dynamics()
    obs_times = jnp.array([0.0, 1.0, 2.0])
    obs_values = jnp.array([0.1, 0.1, 0.1])
    state_path_params = jnp.array(0.1)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                state_path_params=state_path_params,
            )

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

    metadata = dsx.prepare_missing_observation_metadata(
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
    assert jnp.array_equal(metadata.free_flat_indices, jnp.array([1, 2, 4, 5]))
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
    assert "f_state_path_params_lp" in tr
    assert "f_state_path_param_times" in tr
    assert "f_state_path" in tr
    assert "f_state_path_times" in tr
    assert "f_joint_log_prob" in tr
    assert jnp.array_equal(tr["f_state_path_params"]["value"], state_path_params)
    assert jnp.array_equal(tr["f_state_path"]["value"], state_path_params)

    base_dist = (
        dist.Normal(0.0, 1.0)
        .expand(state_path_params.shape)
        .to_event(len(state_path_params.shape))
    )
    expected = dsx.log_prob(
        dynamics,
        state_path_params=state_path_params,
        state_path_param_times=obs_times,
        obs_times=obs_times,
        obs_values=obs_values,
    ) - base_dist.log_prob(state_path_params)
    assert jnp.allclose(tr["f_state_path_params_lp"]["fn"].log_factor, expected)


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

    base_dist = (
        dist.Normal(0.0, 1.0)
        .expand(state_path_params.shape)
        .to_event(len(state_path_params.shape))
    )
    expected = _manual_discrete_state_log_prob(
        dynamics, expected_states, obs_times
    ) - base_dist.log_prob(state_path_params)
    assert jnp.allclose(tr["f_state_path_params_lp"]["fn"].log_factor, expected)


def test_latent_path_builder_sample_can_draw_latents_when_unspecified():
    dynamics = _make_discrete_dynamics()
    obs_times = jnp.array([0.0, 1.0])
    obs_values = jnp.array([0.2, -0.1])

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
    assert tr["f_state_path_params"]["value"].shape == (len(obs_times),)


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
    latent_path_layout = dsx.prepare_latent_path_layout(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )

    def conditioned_model(obs_times=None, obs_values=None):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                latent_path_layout=latent_path_layout,
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
    latent_path_layout = dsx.prepare_latent_path_layout(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )

    def conditioned_model(obs_times=None, obs_values=None):
        with dsx.LatentPathBuilder():
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                latent_path_layout=latent_path_layout,
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
    metadata = dsx.prepare_missing_observation_metadata(
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
    base_dist_x = dist.Normal(0.0, 1.0).expand(state_path_params.shape).to_event(2)
    base_dist_y = dist.Normal(0.0, 1.0).expand(missing_obs_values.shape).to_event(1)
    expected = (
        dsx.log_prob(
            dynamics,
            state_path_params=state_path_params,
            state_path_param_times=obs_times,
            obs_times=obs_times,
            obs_values=obs_values,
            missing_observation_strategy="augment",
            missing_obs_values=missing_obs_values,
            missing_obs_metadata=metadata,
        )
        - base_dist_x.log_prob(state_path_params)
        - base_dist_y.log_prob(missing_obs_values)
    )

    assert "f_missing_obs_values" in tr
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
    assert jnp.allclose(tr["f_state_path_params_lp"]["fn"].log_factor, expected)


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
    latent_path_layout = dsx.prepare_latent_path_layout(
        dynamics_true,
        obs_times=obs_times,
        obs_values=obs_values,
        missing_observation_strategy="auto",
    )

    def conditioned_model(obs_times=None, obs_values=None):
        alpha = numpyro.sample("alpha", dist.Uniform(0.0, 0.9))
        dynamics = _make_student_t_discrete_dynamics(alpha=alpha)
        with dsx.LatentPathBuilder(missing_observation_strategy="auto"):
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                latent_path_layout=latent_path_layout,
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
