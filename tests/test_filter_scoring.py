import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, substitute, trace
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx.evaluation.configs import ObservationScoringConfig
from dynestyx.evaluation.handlers import Evaluation
from dynestyx.evaluation.observation_scoring import compute_observation_scores
from dynestyx.evaluation.scoring import (
    DawidSebastianiScore,
    EnergyScore,
    GaussianLogProbScore,
    ObservationWiseCRPSScore,
)
from dynestyx.inference.configs.filter import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
    KFConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.inference.integrations.cd_dynamax.continuous_filter import (
    compute_continuous_filter,
)
from dynestyx.inference.observation_predictions import (
    _observation_noise_covariance_sequence,
    extract_continuous_filter_predictions,
)
from dynestyx.models.observations import GaussianObservation, LinearGaussianObservation
from dynestyx.simulation import SDESimulator
from dynestyx.simulation.discrete import DiscreteTimeSimulator
from tests.test_utils import assert_tree_all_finite

TRUE_RHO = 1.25


def evaluate_continuous_filter_output(
    posterior,
    *,
    dynamics,
    filter_config,
    obs_times,
    obs_values,
    ctrl_values,
    scoring_config=None,
    plate_shapes=(),
):
    predictions = extract_continuous_filter_predictions(
        posterior,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        ctrl_values=ctrl_values,
        plate_shapes=plate_shapes,
    )
    if scoring_config is None:
        return posterior, predictions, {}
    result = dsx.ConditionedResult(
        times=obs_times,
        predicted_observations=predictions,
    )
    return (
        posterior,
        predictions,
        compute_observation_scores(
            filtered_result=result,
            obs_values=obs_values,
            scoring_config=scoring_config,
            plate_shapes=plate_shapes,
        ),
    )


def _make_continuous_lti_dynamics(rho):
    state_dim = 2
    A = jnp.array([[-1.0, 0.0], [rho, -1.0]])
    L = jnp.eye(state_dim)
    H = jnp.array([[0.0, 1.0]])
    R = jnp.array([[1.0]])
    B = jnp.array([[0.0], [5.0]])
    return dsx.LTI_continuous(A=A, L=L, H=H, R=R, B=B)


def _continuous_lti_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
):
    rho = numpyro.sample("rho", dist.Uniform(0.0, 5.0))
    dynamics = _make_continuous_lti_dynamics(rho)
    dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
    )


def _make_observations():
    obs_times = jnp.linspace(0.0, 0.5, 6)
    ctrl_times = obs_times
    ctrl_values = jnp.sin(obs_times)[:, None]
    with SDESimulator(
        n_simulations=1,
        simulator_config=dsx.SDESimulatorConfig(source="em_scan"),
    ):
        samples = Predictive(
            _continuous_lti_model,
            params={"rho": jnp.array(TRUE_RHO)},
            num_samples=1,
            exclude_deterministic=False,
        )(
            jr.PRNGKey(0),
            predict_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
    return obs_times, samples["f_observations"][0, 0], ctrl_times, ctrl_values


def test_observation_noise_covariance_sequence_uses_constant_structured_R():
    obs_times = jnp.linspace(0.0, 0.5, 6)
    R = jnp.array([[1.0]])
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.5 * x,
            diffusion=dsx.ScalarDiffusion(0.1, bm_dim=1),
        ),
        observation_model=GaussianObservation(
            h=lambda x, u, t: x,
            R=R,
        ),
    )

    covs = _observation_noise_covariance_sequence(
        dynamics,
        obs_times=obs_times,
        ctrl_values=None,
        plate_shapes=(),
    )
    assert jnp.allclose(covs, jnp.broadcast_to(R[None, :, :], covs.shape))


def test_observation_noise_covariance_sequence_falls_back_for_callable_R():
    obs_times = jnp.linspace(0.0, 0.5, 6)
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.5 * x,
            diffusion=dsx.ScalarDiffusion(0.1, bm_dim=1),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.eye(1),
            R=lambda t: jnp.array([[1.0 + t]]),
        ),
    )

    covs = _observation_noise_covariance_sequence(
        dynamics,
        obs_times=obs_times,
        ctrl_values=None,
        plate_shapes=(),
    )
    expected = (1.0 + obs_times)[:, None, None]
    assert jnp.allclose(covs, expected)


def _run_conditioned_trace(
    filter_config, scoring_config, *, obs_times, obs_values, ctrl_times, ctrl_values
):
    with trace() as tr, seed(rng_seed=jr.PRNGKey(99)):
        with substitute(data={"rho": jnp.array(TRUE_RHO)}):
            if scoring_config is None:
                with Filter(filter_config=filter_config):
                    _continuous_lti_model(
                        obs_times=obs_times,
                        obs_values=obs_values,
                        ctrl_times=ctrl_times,
                        ctrl_values=ctrl_values,
                    )
            else:
                with Evaluation(observation_scoring_config=scoring_config):
                    with Filter(filter_config=filter_config):
                        _continuous_lti_model(
                            obs_times=obs_times,
                            obs_values=obs_values,
                            ctrl_times=ctrl_times,
                            ctrl_values=ctrl_values,
                        )
    return tr


@pytest.mark.parametrize(
    ("config_name", "filter_config"),
    [
        ("kf", ContinuousTimeKFConfig()),
        ("ekf", ContinuousTimeEKFConfig()),
        ("ukf", ContinuousTimeUKFConfig()),
        (
            "enkf",
            ContinuousTimeEnKFConfig(
                n_particles=16,
                crn_seed=jr.PRNGKey(7),
            ),
        ),
    ],
)
def test_continuous_filter_scoring_sites_match_pure_backend_outputs(
    config_name,
    filter_config,
):
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    scoring_config = ObservationScoringConfig(
        rules=(
            GaussianLogProbScore(),
            DawidSebastianiScore(),
            ObservationWiseCRPSScore(),
        )
    )
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    key = (
        filter_config.crn_seed if filter_config.crn_seed is not None else jr.PRNGKey(3)
    )
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    _, predictions, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )
    assert predictions is not None
    assert predictions.mean is not None
    assert predictions.cov is not None

    tr = _run_conditioned_trace(
        filter_config,
        scoring_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    assert_tree_all_finite(
        {
            "gaussian_log_prob": tr["f_gaussian_log_prob"]["value"],
            "dawid_sebastiani": tr["f_dawid_sebastiani"]["value"],
            "observation_wise_crps": tr["f_observation_wise_crps"]["value"],
        },
        where=f"{config_name} scoring outputs",
    )
    assert jnp.allclose(
        tr["f_gaussian_log_prob"]["value"],
        score_arrays["gaussian_log_prob"],
    )
    assert jnp.allclose(
        tr["f_dawid_sebastiani"]["value"],
        score_arrays["dawid_sebastiani"],
    )
    assert jnp.allclose(
        tr["f_observation_wise_crps"]["value"],
        score_arrays["observation_wise_crps"],
    )

    assert "f_predicted_observations_mean" in tr
    assert "f_predicted_observations_cov" in tr
    if isinstance(filter_config, ContinuousTimeEnKFConfig):
        assert "f_predicted_observations_ensemble" in tr
    else:
        assert "f_predicted_observations_ensemble" not in tr


@pytest.mark.parametrize(
    ("config_name", "filter_config"),
    [
        (
            "kf",
            ContinuousTimeKFConfig(
                record_predicted_observations_mean=True,
                record_predicted_observations_cov=True,
            ),
        ),
        (
            "ekf",
            ContinuousTimeEKFConfig(
                record_predicted_observations_mean=True,
                record_predicted_observations_cov=True,
            ),
        ),
        (
            "ukf",
            ContinuousTimeUKFConfig(
                record_predicted_observations_mean=True,
                record_predicted_observations_cov=True,
            ),
        ),
        (
            "enkf",
            ContinuousTimeEnKFConfig(
                n_particles=16,
                crn_seed=jr.PRNGKey(7),
                record_predicted_observations_mean=True,
                record_predicted_observations_cov=True,
                record_predicted_observations_ensemble=True,
            ),
        ),
    ],
)
def test_continuous_filter_predicted_observation_recording_sites_match_backend_outputs(
    config_name,
    filter_config,
):
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    key = (
        filter_config.crn_seed if filter_config.crn_seed is not None else jr.PRNGKey(3)
    )
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    _, predictions, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=None,
    )
    assert predictions is not None
    assert score_arrays == {}
    assert predictions.mean is not None
    assert predictions.cov is not None

    tr = _run_conditioned_trace(
        filter_config,
        None,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    assert_tree_all_finite(
        {
            "pred_mean": tr["f_predicted_observations_mean"]["value"],
            "pred_cov": tr["f_predicted_observations_cov"]["value"],
        },
        where=f"{config_name} predicted observation recordings",
    )
    assert jnp.allclose(
        tr["f_predicted_observations_mean"]["value"],
        predictions.mean,
    )
    assert jnp.allclose(
        tr["f_predicted_observations_cov"]["value"],
        predictions.cov,
    )

    if isinstance(filter_config, ContinuousTimeEnKFConfig):
        assert "f_predicted_observations_ensemble" in tr
        assert predictions.ensemble is not None
        assert jnp.allclose(
            tr["f_predicted_observations_ensemble"]["value"],
            predictions.ensemble,
        )


def test_scoring_config_can_compute_without_recording_sites():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeKFConfig(
        record_predicted_observations_mean=False,
        record_predicted_observations_cov=False,
        record_predicted_observations_ensemble=False,
    )
    scoring_config = ObservationScoringConfig(
        rules=(GaussianLogProbScore(),),
        record_as_numpyro_sites=False,
    )
    tr = _run_conditioned_trace(
        filter_config,
        scoring_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    assert "f_gaussian_log_prob" not in tr


def test_scoring_still_validates_when_score_sites_are_disabled():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeKFConfig()
    scoring_config = ObservationScoringConfig(
        rules=(EnergyScore(beta=1.0),),
        record_as_numpyro_sites=False,
        sample_source="backend_ensemble",
    )
    with pytest.raises(
        NotImplementedError, match="predicted_observations.obs_ensemble"
    ):
        _run_conditioned_trace(
            filter_config,
            scoring_config,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def test_scoring_does_not_require_predicted_observation_recording():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeKFConfig(
        record_predicted_observations_mean=False,
        record_predicted_observations_cov=False,
        record_predicted_observations_ensemble=False,
    )
    scoring_config = ObservationScoringConfig(
        rules=(GaussianLogProbScore(),),
        record_as_numpyro_sites=True,
    )
    tr = _run_conditioned_trace(
        filter_config,
        scoring_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    assert "f_gaussian_log_prob" in tr
    assert "f_predicted_observations_mean" not in tr
    assert "f_predicted_observations_cov" not in tr


def test_energy_score_records_for_enkf():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeEnKFConfig(
        n_particles=16,
        crn_seed=jr.PRNGKey(11),
    )
    scoring_config = ObservationScoringConfig(
        rules=(EnergyScore(beta=1.0), EnergyScore(beta=1.5)),
    )
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=filter_config.crn_seed,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    _, predictions, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )
    assert predictions is not None
    tr = _run_conditioned_trace(
        filter_config,
        scoring_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    assert "f_energy_score" in tr
    assert "f_energy_score_beta_1_5" in tr
    assert jnp.allclose(tr["f_energy_score"]["value"], score_arrays["energy_score"])
    assert jnp.allclose(
        tr["f_energy_score_beta_1_5"]["value"],
        score_arrays["energy_score_beta_1_5"],
    )


def test_energy_score_vectorized_and_scan_match():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeEnKFConfig(
        n_particles=16,
        crn_seed=jr.PRNGKey(13),
    )
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=filter_config.crn_seed,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    _, predictions, _ = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=ObservationScoringConfig(rules=(EnergyScore(beta=1.5),)),
    )
    assert predictions is not None
    assert predictions.obs_ensemble is not None

    vectorized_score = EnergyScore(
        beta=1.5,
        vectorized_pairwise=True,
    ).compute(
        obs_values=obs_values,
        pred_ensemble=predictions.obs_ensemble,
    )
    scan_score = EnergyScore(
        beta=1.5,
        vectorized_pairwise=False,
    ).compute(
        obs_values=obs_values,
        pred_ensemble=predictions.obs_ensemble,
    )
    assert jnp.allclose(vectorized_score, scan_score)


def test_enkf_energy_score_defaults_to_predictive_observation_ensemble():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeEnKFConfig(
        n_particles=16,
        crn_seed=jr.PRNGKey(17),
    )
    scoring_config = ObservationScoringConfig(
        rules=(EnergyScore(beta=1.0),),
        sample_seed=9,
    )
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=filter_config.crn_seed,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    _, predictions, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )
    assert predictions is not None
    assert predictions.ensemble is not None
    assert predictions.obs_ensemble is not None

    expected_score = EnergyScore(beta=1.0).compute(
        obs_values=obs_values,
        pred_ensemble=predictions.obs_ensemble,
    )
    latent_score = EnergyScore(beta=1.0).compute(
        obs_values=obs_values,
        pred_ensemble=predictions.ensemble,
    )

    assert jnp.allclose(score_arrays["energy_score"], expected_score)
    assert not jnp.allclose(score_arrays["energy_score"], latent_score)


def test_kf_gaussian_scores_use_predictive_observation_covariance():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeKFConfig()
    scoring_config = ObservationScoringConfig(rules=(GaussianLogProbScore(),))
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=jr.PRNGKey(23),
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    _, predictions, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )
    assert predictions is not None
    assert predictions.mean is not None
    assert predictions.cov is not None
    assert predictions.obs_cov is not None

    expected_score = GaussianLogProbScore().compute(
        obs_values=obs_values,
        pred_mean=predictions.mean,
        pred_cov=predictions.obs_cov,
    )
    latent_score = GaussianLogProbScore().compute(
        obs_values=obs_values,
        pred_mean=predictions.mean,
        pred_cov=predictions.cov,
    )
    assert jnp.allclose(score_arrays["gaussian_log_prob"], expected_score)
    assert not jnp.allclose(score_arrays["gaussian_log_prob"], latent_score)


def test_gaussian_scores_ignore_ensemble_sample_source_when_unused():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeKFConfig()
    scoring_config = ObservationScoringConfig(
        rules=(GaussianLogProbScore(),),
        sample_source="backend_ensemble",
    )
    tr = _run_conditioned_trace(
        filter_config,
        scoring_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    assert "f_gaussian_log_prob" in tr


def test_unsupported_skip_can_keep_gaussian_scores_without_ensemble_source():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeKFConfig()
    scoring_config = ObservationScoringConfig(
        rules=(GaussianLogProbScore(), EnergyScore(beta=1.0)),
        sample_source="backend_ensemble",
        unsupported="skip",
    )
    tr = _run_conditioned_trace(
        filter_config,
        scoring_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    assert "f_gaussian_log_prob" in tr
    assert "f_energy_score" not in tr


def test_backend_observation_ensemble_source_is_rejected_when_unavailable():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    with pytest.raises(
        NotImplementedError,
        match="predicted_observations.obs_ensemble",
    ):
        _run_conditioned_trace(
            ContinuousTimeKFConfig(),
            ObservationScoringConfig(
                rules=(EnergyScore(beta=1.0),),
                sample_source="backend_ensemble",
            ),
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def test_backend_observation_ensemble_source_is_used_when_available():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeEnKFConfig(
        n_particles=16,
        crn_seed=jr.PRNGKey(31),
    )
    scoring_config = ObservationScoringConfig(
        rules=(EnergyScore(beta=1.0),),
        sample_source="backend_ensemble",
    )
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=filter_config.crn_seed,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    latent_ensemble = jnp.asarray(filtered.y_ens_pred)
    backend_obs_ensemble = latent_ensemble + 0.25
    filtered = filtered._replace(
        y_obs_ens_pred=backend_obs_ensemble,
    )
    _, predictions, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )
    assert predictions is not None
    assert predictions.obs_ensemble is not None
    assert jnp.allclose(predictions.obs_ensemble, backend_obs_ensemble)

    expected_score = EnergyScore(beta=1.0).compute(
        obs_values=obs_values,
        pred_ensemble=backend_obs_ensemble,
    )
    assert jnp.allclose(score_arrays["energy_score"], expected_score)


def test_auto_prefers_backend_observation_ensemble():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeEnKFConfig(
        n_particles=16,
        crn_seed=jr.PRNGKey(37),
    )
    scoring_config = ObservationScoringConfig(
        rules=(EnergyScore(beta=1.0),),
        sample_seed=5,
    )
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=filter_config.crn_seed,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    latent_ensemble = jnp.asarray(filtered.y_ens_pred)
    backend_obs_ensemble = latent_ensemble + 0.5
    filtered = filtered._replace(
        y_obs_ens_pred=backend_obs_ensemble,
    )
    _, predictions, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )
    assert predictions is not None
    assert predictions.obs_ensemble is not None
    assert predictions.ensemble is not None
    assert predictions.noise_cov is not None

    expected_score = EnergyScore(beta=1.0).compute(
        obs_values=obs_values,
        pred_ensemble=backend_obs_ensemble,
    )
    sampled_score = EnergyScore(beta=1.0).compute(
        obs_values=obs_values,
        pred_ensemble=predictions.ensemble
        + jnp.moveaxis(
            dist.MultivariateNormal(
                loc=jnp.zeros_like(predictions.ensemble[..., 0, :]),
                covariance_matrix=predictions.noise_cov,
            ).sample(jr.PRNGKey(scoring_config.sample_seed), sample_shape=(16,)),
            0,
            -2,
        ),
    )
    assert jnp.allclose(score_arrays["energy_score"], expected_score)
    assert not jnp.allclose(score_arrays["energy_score"], sampled_score)


def test_energy_score_can_sample_gaussian_ensemble_for_kf():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    filter_config = ContinuousTimeKFConfig()
    scoring_config = ObservationScoringConfig(
        rules=(
            GaussianLogProbScore(),
            EnergyScore(beta=1.0, n_samples=64),
        ),
        sample_seed=5,
    )
    dynamics = _make_continuous_lti_dynamics(TRUE_RHO)
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key=jr.PRNGKey(13),
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    _, _, score_arrays = evaluate_continuous_filter_output(
        filtered,
        dynamics=dynamics,
        filter_config=filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
        scoring_config=scoring_config,
    )
    tr = _run_conditioned_trace(
        filter_config,
        scoring_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )
    assert "f_gaussian_log_prob" in tr
    assert "f_energy_score" in tr
    assert jnp.allclose(tr["f_energy_score"]["value"], score_arrays["energy_score"])


def test_energy_score_requires_n_samples_without_ensemble():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    with pytest.raises(
        NotImplementedError,
        match="together with `n_samples`",
    ):
        _run_conditioned_trace(
            ContinuousTimeKFConfig(),
            ObservationScoringConfig(rules=(EnergyScore(beta=1.0),)),
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def test_continuous_dpf_scoring_is_not_supported_yet():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    with pytest.raises(
        ValueError,
        match="filtered_result.predicted_observations",
    ):
        _run_conditioned_trace(
            ContinuousTimeDPFConfig(n_particles=16),
            ObservationScoringConfig(rules=(GaussianLogProbScore(),)),
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def test_condition_returns_attached_evaluation_result_without_registering_sites():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    scoring_config = ObservationScoringConfig(rules=(GaussianLogProbScore(),))

    with trace() as tr, seed(rng_seed=jr.PRNGKey(39)):
        with Evaluation(observation_scoring_config=scoring_config):
            with Filter(filter_config=ContinuousTimeKFConfig()):
                result = dsx.condition(
                    "f",
                    _make_continuous_lti_dynamics(TRUE_RHO),
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                )

    assert result.evaluation_result is not None
    assert "gaussian_log_prob" in result.evaluation_result.observation_scores
    assert "f_gaussian_log_prob" not in tr
    assert "f_marginal_log_likelihood" not in tr


def test_evaluation_composes_with_filter_and_simulator_registration():
    obs_times, obs_values, _, _ = _make_observations()
    predict_times = jnp.linspace(0.0, 0.75, 8)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(40)):
        with Evaluation(
            observation_scoring_config=ObservationScoringConfig(
                rules=(GaussianLogProbScore(),)
            )
        ):
            with SDESimulator(
                n_simulations=1,
                simulator_config=dsx.SDESimulatorConfig(source="em_scan"),
            ):
                with Filter(filter_config=ContinuousTimeKFConfig()):
                    result = dsx.sample(
                        "f",
                        _make_continuous_lti_dynamics(TRUE_RHO),
                        obs_times=obs_times,
                        obs_values=obs_values,
                        predict_times=predict_times,
                    )

    assert result.evaluation_result is not None
    assert "f_gaussian_log_prob" in tr
    assert "f_marginal_log_likelihood" in tr
    assert "f_predicted_states" in tr


def test_evaluation_explains_handler_order():
    obs_times, obs_values, _, _ = _make_observations()
    with pytest.raises(ValueError, match="Place Evaluation outside Filter"):
        with Evaluation(
            observation_scoring_config=ObservationScoringConfig(
                rules=(GaussianLogProbScore(),)
            )
        ):
            dsx.condition(
                "f",
                _make_continuous_lti_dynamics(TRUE_RHO),
                obs_times=obs_times,
                obs_values=obs_values,
            )


def test_evaluation_explains_disabled_prediction_collection():
    obs_times, obs_values, _, _ = _make_observations()
    with pytest.raises(ValueError, match="include_predicted_observations=True"):
        with Evaluation(
            observation_scoring_config=ObservationScoringConfig(
                rules=(GaussianLogProbScore(),)
            )
        ):
            with Filter(
                filter_config=ContinuousTimeKFConfig(
                    include_predicted_observations=False
                )
            ):
                dsx.condition(
                    "f",
                    _make_continuous_lti_dynamics(TRUE_RHO),
                    obs_times=obs_times,
                    obs_values=obs_values,
                )


def test_evaluation_rejects_missing_observations_through_simulator():
    obs_times = jnp.arange(3.0)
    obs_values = jnp.array([[0.0], [jnp.nan], [0.2]])
    dynamics = dsx.LTI_discrete(
        A=jnp.eye(1),
        Q=0.1 * jnp.eye(1),
        H=jnp.eye(1),
        R=0.2 * jnp.eye(1),
    )

    with pytest.raises(
        ValueError,
        match="Observation scoring does not yet support missing obs_values",
    ):
        with Evaluation(
            observation_scoring_config=ObservationScoringConfig(
                rules=(GaussianLogProbScore(),)
            )
        ):
            with DiscreteTimeSimulator():
                with Filter(filter_config=KFConfig(filter_source="cuthbert")):
                    dsx.condition(
                        "f",
                        dynamics,
                        obs_times=obs_times,
                        obs_values=obs_values,
                    )


def test_plate_batched_scoring_and_conditioned_result_outputs():
    obs_times, obs_values, ctrl_times, ctrl_values = _make_observations()
    plate_size = 2
    plate_obs_values = jnp.broadcast_to(
        obs_values,
        (plate_size, *obs_values.shape),
    )
    filter_config = ContinuousTimeEKFConfig(
        record_predicted_observations_mean=True,
        record_predicted_observations_cov=True,
    )
    scoring_config = ObservationScoringConfig(
        rules=(GaussianLogProbScore(), ObservationWiseCRPSScore()),
    )

    def plate_model(obs_times, obs_values, ctrl_times, ctrl_values):
        with dsx.plate("trajectories", plate_size):
            return dsx.sample(
                "f",
                _make_continuous_lti_dynamics(TRUE_RHO),
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(41)):
        with Evaluation(observation_scoring_config=scoring_config):
            with Filter(filter_config=filter_config):
                result = plate_model(
                    obs_times,
                    plate_obs_values,
                    ctrl_times,
                    ctrl_values,
                )

    assert result.predicted_observations is not None
    assert result.predicted_observations.mean is not None
    assert result.predicted_observations.mean.shape == (
        plate_size,
        obs_times.shape[0],
        1,
    )
    assert result.evaluation_result is not None
    assert result.evaluation_result.observation_scores["gaussian_log_prob"].shape == (
        plate_size,
        obs_times.shape[0],
        1,
    )
    assert tr["f_observation_wise_crps"]["value"].shape == (
        plate_size,
        obs_times.shape[0],
        1,
    )


def test_plate_batched_scoring_broadcasts_shared_observations():
    obs_times, obs_values, _, _ = _make_observations()
    plate_size = 2
    plate_obs_times = jnp.broadcast_to(obs_times, (plate_size, obs_times.shape[0]))

    def plate_model(obs_times, obs_values):
        with dsx.plate("trajectories", plate_size):
            return dsx.sample(
                "f",
                _make_continuous_lti_dynamics(TRUE_RHO),
                obs_times=obs_times,
                obs_values=obs_values,
            )

    with trace(), seed(rng_seed=jr.PRNGKey(43)):
        with Evaluation(
            observation_scoring_config=ObservationScoringConfig(
                rules=(GaussianLogProbScore(),),
            )
        ):
            with Filter(filter_config=ContinuousTimeEKFConfig()):
                result = plate_model(plate_obs_times, obs_values)

    assert result.evaluation_result is not None
    assert result.evaluation_result.observation_scores["gaussian_log_prob"].shape == (
        plate_size,
        obs_times.shape[0],
        1,
    )
