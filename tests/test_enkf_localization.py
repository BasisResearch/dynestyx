"""Focused tests for structured Cuthbert EnKF localization."""

import dataclasses

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from cuthbertlib.ensemble_kalman.localization import (
    construct_tapered_chol_innovation_covariance,
    gaspari_cohn,
    gaussian,
)

import dynestyx as dsx
from dynestyx.inference.configs.filter import (
    ContinuousTimeEnKFConfig,
    EnKFConfig,
    EnKFLocalizationConfig,
    EnKFLocalizationFunctions,
)
from dynestyx.inference.configs.smoother import EnRTSSmootherConfig
from dynestyx.inference.enkf_localization import resolve_enkf_localization
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    build_cuthbert_filter,
    compute_cuthbert_filter,
)
from dynestyx.inference.integrations.cuthbert.discrete_smoother import (
    compute_cuthbert_smoother,
)
from dynestyx.inference.observation_predictions import extract_filter_predictions


def _problem():
    dynamics = dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.array([0.1, -0.2, 0.3]),
            covariance_matrix=jnp.array(
                [[0.8, 0.1, 0.0], [0.1, 0.6, 0.05], [0.0, 0.05, 0.5]]
            ),
        ),
        state_evolution=dsx.LinearGaussianStateEvolution(
            A=jnp.array([[0.85, 0.1, 0.0], [-0.05, 0.9, 0.08], [0.02, 0.0, 0.8]]),
            cov=0.06 * jnp.eye(3),
        ),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.array([[1.0, 0.0, 0.2], [0.0, 1.0, -0.1]]),
            R=jnp.array([[0.25, 0.03], [0.03, 0.2]]),
        ),
    )
    obs_times = jnp.arange(4.0)
    obs_values = jnp.array([[0.2, -0.1], [0.0, 0.25], [0.3, 0.1], [-0.15, 0.4]])
    state_observation_distances = jnp.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    observation_distances = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    return (
        dynamics,
        obs_times,
        obs_values,
        state_observation_distances,
        observation_distances,
    )


def _run(config, *, obs_values=None):
    dynamics, obs_times, default_values, _, _ = _problem()
    values = default_values if obs_values is None else obs_values
    return compute_cuthbert_filter(
        dynamics,
        config,
        jr.PRNGKey(7),
        obs_times=obs_times,
        obs_values=values,
    )


@pytest.mark.parametrize(
    ("name", "covariance_fn"),
    [("gaspari_cohn", gaspari_cohn), ("gaussian", gaussian)],
)
def test_builtin_tapers_modify_cross_and_marginal_covariances(name, covariance_fn):
    _, _, _, cross_distances, observation_distances = _problem()
    scale = 2.5
    resolved = resolve_enkf_localization(
        EnKFLocalizationConfig(
            state_observation_distances=cross_distances,
            observation_distances=observation_distances,
            taper=name,
            taper_scale=scale,
        ),
        state_dim=3,
        observation_dim=2,
    )

    cross_covariance = jnp.arange(6.0).reshape(3, 2)
    expected_cross = cross_covariance * covariance_fn(cross_distances, scale)
    assert resolved.modify_cross_covariance is not None
    assert jnp.allclose(
        resolved.modify_cross_covariance(cross_covariance, None),
        expected_cross,
    )

    expected_taper = covariance_fn(observation_distances, scale)
    assert resolved.observation_taper is not None
    assert jnp.allclose(resolved.observation_taper, expected_taper)
    normalized_deviations = jnp.array([[0.5, -0.2, 0.1], [0.3, 0.4, -0.1]])
    chol_noise = jnp.linalg.cholesky(jnp.array([[0.3, 0.02], [0.02, 0.2]]))
    assert resolved.construct_chol_innovation_covariance is not None
    chol_innovation = resolved.construct_chol_innovation_covariance(
        normalized_deviations,
        chol_noise,
        None,
    )
    expected_covariance = (
        expected_taper * (normalized_deviations @ normalized_deviations.T)
        + chol_noise @ chol_noise.T
    )
    assert jnp.allclose(
        chol_innovation @ chol_innovation.T,
        expected_covariance,
        rtol=2e-5,
        atol=2e-5,
    )


def test_custom_taper_is_evaluated_only_for_two_distance_matrices():
    dynamics, obs_times, obs_values, cross_distances, observation_distances = _problem()
    calls = []

    def custom_taper(distances):
        calls.append(distances.shape)
        return jnp.exp(-0.5 * distances**2)

    config = EnKFConfig(
        n_particles=10,
        perturb_measurements=False,
        localization=EnKFLocalizationConfig(
            state_observation_distances=cross_distances,
            observation_distances=observation_distances,
            taper=custom_taper,
        ),
    )
    _, states = compute_cuthbert_filter(
        dynamics,
        config,
        jr.PRNGKey(8),
        obs_times=obs_times,
        obs_values=obs_values,
    )
    predictions = extract_filter_predictions(
        states,
        dynamics=dynamics,
        filter_config=config,
        obs_times=obs_times,
        ctrl_values=None,
    )

    assert calls == [(3, 2), (2, 2)]
    assert predictions is not None
    assert predictions.cov is not None
    assert predictions.cov.shape == (4, 2, 2)


def test_direct_callbacks_are_forwarded_unchanged_and_receive_model_inputs():
    dynamics, obs_times, _, _, _ = _problem()

    def taper_at_time(time):
        correlation = 0.25 + 0.05 * jnp.tanh(time)
        return jnp.array([[1.0, correlation], [correlation, 1.0]])

    def modify_cross(cross_covariance, model_inputs):
        return cross_covariance * (1.0 + 0.02 * model_inputs.time)

    def construct_innovation(
        normalized_observation_deviations,
        chol_observation_covariance,
        model_inputs,
    ):
        return construct_tapered_chol_innovation_covariance(
            normalized_observation_deviations,
            jnp.linalg.cholesky(taper_at_time(model_inputs.time)),
            chol_observation_covariance,
        )

    def modify_prediction(predicted_observation_covariance, model_inputs):
        return predicted_observation_covariance * taper_at_time(model_inputs.time)

    localization = EnKFLocalizationFunctions(
        modify_cross_covariance=modify_cross,
        construct_chol_innovation_covariance=construct_innovation,
        modify_predicted_observation_covariance=modify_prediction,
    )
    config = EnKFConfig(n_particles=10, localization=localization)
    filter_obj, _ = build_cuthbert_filter(
        dynamics,
        config,
        jr.PRNGKey(9),
        want_parallel=False,
    )

    _, states = _run(config)
    assert jnp.all(jnp.isfinite(states.ensemble))
    model_inputs_0 = jax.tree.map(lambda leaf: leaf[0], states.model_inputs)
    cross_covariance = jnp.arange(6.0).reshape(3, 2)
    wrapped_cross = filter_obj.filter_combine.keywords["modify_cross_covariance"]
    assert jnp.allclose(
        wrapped_cross(cross_covariance, model_inputs_0),
        modify_cross(cross_covariance, model_inputs_0),
    )
    normalized_deviations = jnp.array([[0.2, -0.1], [0.3, 0.05]])
    chol_noise = jnp.linalg.cholesky(jnp.array([[0.25, 0.03], [0.03, 0.2]]))
    wrapped_constructor = filter_obj.filter_combine.keywords[
        "construct_chol_innovation_covariance"
    ]
    assert jnp.allclose(
        wrapped_constructor(normalized_deviations, chol_noise, model_inputs_0),
        construct_innovation(normalized_deviations, chol_noise, model_inputs_0),
    )
    predictions = extract_filter_predictions(
        states,
        dynamics=dynamics,
        filter_config=config,
        obs_times=obs_times,
        ctrl_values=None,
    )
    assert predictions is not None
    assert predictions.cov is not None
    raw_ensemble = jnp.einsum(
        "ij,tnj->tni",
        dynamics.observation_model.H,
        states.predicted_ensemble,
    )
    raw_deviations = raw_ensemble - jnp.mean(raw_ensemble, axis=-2, keepdims=True)
    raw_covariance = jnp.einsum("tni,tnj->tij", raw_deviations, raw_deviations) / (
        config.n_particles - 1
    )
    expected_tapers = jax.vmap(taper_at_time)(obs_times)
    assert jnp.allclose(predictions.cov, raw_covariance * expected_tapers)


def test_marginal_localization_scores_match_filter_likelihood_and_keeps_raw_ensemble():
    dynamics, obs_times, obs_values, cross_distances, observation_distances = _problem()
    config = EnKFConfig(
        n_particles=14,
        perturb_measurements=False,
        crn_seed=jr.PRNGKey(10),
        localization=EnKFLocalizationConfig(
            state_observation_distances=cross_distances,
            observation_distances=observation_distances,
            taper="gaussian",
            taper_scale=1.3,
        ),
    )
    marginal_loglik, states = compute_cuthbert_filter(
        dynamics,
        config,
        config.crn_seed,
        obs_times=obs_times,
        obs_values=obs_values,
    )
    predictions = extract_filter_predictions(
        states,
        dynamics=dynamics,
        filter_config=config,
        obs_times=obs_times,
        ctrl_values=None,
    )
    assert predictions is not None
    assert predictions.mean is not None
    assert predictions.cov is not None
    assert predictions.obs_cov is not None
    assert predictions.ensemble is not None

    H = dynamics.observation_model.H
    expected_raw_ensemble = jnp.einsum("ij,tnj->tni", H, states.predicted_ensemble)
    raw_mean = jnp.mean(expected_raw_ensemble, axis=-2)
    raw_deviations = expected_raw_ensemble - raw_mean[:, None, :]
    raw_covariance = jnp.einsum("tni,tnj->tij", raw_deviations, raw_deviations) / (
        config.n_particles - 1
    )
    expected_taper = gaussian(observation_distances, 1.3)

    assert jnp.allclose(predictions.ensemble, expected_raw_ensemble)
    assert jnp.allclose(predictions.cov, raw_covariance * expected_taper)
    expected_loglik = dist.MultivariateNormal(
        loc=predictions.mean,
        covariance_matrix=predictions.obs_cov,
    ).log_prob(obs_values)
    cumulative_loglik = states.log_normalizing_constant
    per_step_loglik = jnp.diff(
        jnp.concatenate([jnp.zeros_like(cumulative_loglik[:1]), cumulative_loglik])
    )
    assert jnp.allclose(expected_loglik, per_step_loglik, rtol=3e-5, atol=3e-5)
    assert jnp.allclose(jnp.sum(expected_loglik), marginal_loglik, rtol=3e-5)


def test_no_localization_path_is_exactly_unchanged():
    base_config = EnKFConfig(
        n_particles=10,
        perturb_measurements=False,
        crn_seed=jr.PRNGKey(11),
    )
    explicit_none = dataclasses.replace(base_config, localization=None)
    ll_base, states_base = _run(base_config)
    ll_none, states_none = _run(explicit_none)

    assert jnp.array_equal(ll_base, ll_none)
    for base_leaf, none_leaf in zip(
        jax.tree.leaves(states_base),
        jax.tree.leaves(states_none),
        strict=True,
    ):
        assert jnp.array_equal(base_leaf, none_leaf)


def test_localization_supports_missing_observations_and_plate_vmap():
    dynamics, obs_times, obs_values, cross_distances, observation_distances = _problem()
    config = EnKFConfig(
        n_particles=10,
        localization=EnKFLocalizationConfig(
            state_observation_distances=cross_distances,
            observation_distances=observation_distances,
            taper="gaussian",
            taper_scale=1.4,
        ),
    )
    missing_values = obs_values.at[1, 0].set(jnp.nan)
    missing_values = missing_values.at[2].set(jnp.nan)
    missing_loglik, missing_states = compute_cuthbert_filter(
        dynamics,
        config,
        jr.PRNGKey(12),
        obs_times=obs_times,
        obs_values=missing_values,
    )
    assert jnp.isfinite(missing_loglik)
    assert jnp.all(jnp.isfinite(missing_states.ensemble))

    plate_values = jnp.stack([obs_values, obs_values + 0.1])
    keys = jr.split(jr.PRNGKey(13), 2)
    _, plate_states = jax.vmap(
        lambda values, key: compute_cuthbert_filter(
            dynamics,
            config,
            key,
            obs_times=obs_times,
            obs_values=values,
        )
    )(plate_values, keys)
    predictions = extract_filter_predictions(
        plate_states,
        dynamics=dynamics,
        filter_config=config,
        obs_times=jnp.broadcast_to(obs_times, (2, 4)),
        ctrl_values=None,
        plate_shapes=(2,),
    )
    assert predictions is not None
    assert predictions.cov is not None
    assert predictions.ensemble is not None
    assert predictions.cov.shape == (2, 4, 2, 2)
    assert predictions.ensemble.shape == (2, 4, 10, 2)


def test_localization_supports_enrts_forward_filter():
    dynamics, obs_times, obs_values, cross_distances, observation_distances = _problem()
    config = EnRTSSmootherConfig(
        n_particles=10,
        localization=EnKFLocalizationConfig(
            state_observation_distances=cross_distances,
            observation_distances=observation_distances,
            taper="gaspari_cohn",
            taper_scale=2.5,
        ),
    )
    marginal_loglik, states = compute_cuthbert_smoother(
        dynamics,
        config,
        jr.PRNGKey(14),
        obs_times=obs_times,
        obs_values=obs_values,
    )
    forward_config = EnKFConfig(
        n_particles=config.n_particles,
        crn_seed=config.crn_seed,
        perturb_measurements=config.perturb_measurements,
        inflation_delta=config.inflation_delta,
        localization=config.localization,
    )
    forward_loglik, _ = compute_cuthbert_filter(
        dynamics,
        forward_config,
        jr.PRNGKey(14),
        obs_times=obs_times,
        obs_values=obs_values,
    )
    assert jnp.isfinite(marginal_loglik)
    assert jnp.array_equal(marginal_loglik, forward_loglik)
    assert states.ensemble.shape == (4, 10, 3)
    assert jnp.all(jnp.isfinite(states.ensemble))


@pytest.mark.parametrize("custom_taper", [False, True])
def test_gaussian_scale_is_jittable_vmappable_and_differentiable(custom_taper):
    dynamics, obs_times, obs_values, cross_distances, observation_distances = _problem()

    def objective(log_scale):
        scale = jnp.exp(log_scale)
        localization = (
            EnKFLocalizationConfig(
                state_observation_distances=cross_distances,
                observation_distances=observation_distances,
                taper=lambda distances: gaussian(distances, scale),
            )
            if custom_taper
            else EnKFLocalizationConfig(
                state_observation_distances=cross_distances,
                observation_distances=observation_distances,
                taper="gaussian",
                taper_scale=scale,
            )
        )
        config = EnKFConfig(
            n_particles=8,
            perturb_measurements=False,
            localization=localization,
        )
        marginal_loglik, _ = compute_cuthbert_filter(
            dynamics,
            config,
            jr.PRNGKey(15),
            obs_times=obs_times,
            obs_values=obs_values,
        )
        return marginal_loglik

    value = jax.jit(objective)(jnp.array(0.2))
    vmapped = jax.vmap(objective)(jnp.array([-0.1, 0.2, 0.5]))
    gradient = jax.value_and_grad(objective)(jnp.array(0.2))[1]
    step = 2e-2
    finite_difference = (
        objective(jnp.array(0.2 + step)) - objective(jnp.array(0.2 - step))
    ) / (2 * step)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(vmapped))
    assert jnp.isfinite(gradient)
    assert jnp.allclose(gradient, finite_difference, rtol=3e-2, atol=3e-2)


def test_localization_configuration_validation():
    _, _, _, cross_distances, observation_distances = _problem()

    with pytest.raises(ValueError, match="requires a positive scalar"):
        EnKFLocalizationConfig(state_observation_distances=cross_distances)
    with pytest.raises(ValueError, match="requires taper_scale=None"):
        EnKFLocalizationConfig(
            state_observation_distances=cross_distances,
            taper=lambda distances: distances,
            taper_scale=1.0,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        resolve_enkf_localization(
            EnKFLocalizationConfig(
                state_observation_distances=cross_distances,
                taper_scale=0.0,
            ),
            state_dim=3,
            observation_dim=2,
        )
    with pytest.raises(ValueError, match="shape"):
        resolve_enkf_localization(
            EnKFLocalizationConfig(
                state_observation_distances=cross_distances,
                observation_distances=observation_distances,
                taper=lambda distances: jnp.ones((1,)),
            ),
            state_dim=3,
            observation_dim=2,
        )
    with pytest.raises(ValueError, match="positive definite"):
        resolve_enkf_localization(
            EnKFLocalizationConfig(
                state_observation_distances=cross_distances,
                observation_distances=observation_distances,
                taper=jnp.ones_like,
            ),
            state_dim=3,
            observation_dim=2,
        )
    with pytest.raises(ValueError, match="nonnegative"):
        resolve_enkf_localization(
            EnKFLocalizationConfig(
                state_observation_distances=cross_distances.at[0, 0].set(-1.0),
                taper_scale=1.0,
            ),
            state_dim=3,
            observation_dim=2,
        )
    with pytest.raises(ValueError, match="symmetric"):
        resolve_enkf_localization(
            EnKFLocalizationConfig(
                state_observation_distances=cross_distances,
                observation_distances=observation_distances.at[0, 1].set(2.0),
                taper_scale=1.0,
            ),
            state_dim=3,
            observation_dim=2,
        )


def test_direct_callback_pairing_reserved_names_and_continuous_rejection():
    _, _, _, cross_distances, _ = _problem()

    def construct_innovation(
        normalized_observation_deviations,
        chol_observation_covariance,
        model_inputs,
    ):
        del normalized_observation_deviations, model_inputs
        return chol_observation_covariance

    def modify_prediction(predicted_observation_covariance, model_inputs):
        del model_inputs
        return predicted_observation_covariance

    with pytest.raises(ValueError, match="at least one"):
        EnKFLocalizationFunctions()
    with pytest.raises(ValueError, match="supplied together"):
        EnKFLocalizationFunctions(
            construct_chol_innovation_covariance=construct_innovation
        )
    with pytest.raises(ValueError, match="supplied together"):
        EnKFLocalizationFunctions(
            modify_predicted_observation_covariance=modify_prediction
        )
    for hook_name in (
        "modify_cross_covariance",
        "construct_chol_innovation_covariance",
        "modify_predicted_observation_covariance",
    ):
        with pytest.raises(ValueError, match="reserved"):
            EnKFConfig(extra_filter_kwargs={hook_name: lambda value: value})

    localization = EnKFLocalizationConfig(
        state_observation_distances=cross_distances,
        taper_scale=1.0,
    )
    with pytest.raises(ValueError, match="Discretizer"):
        ContinuousTimeEnKFConfig(localization=localization)


def test_direct_callback_outputs_are_shape_and_finiteness_checked():
    def identity_prediction(predicted_observation_covariance, model_inputs):
        del model_inputs
        return predicted_observation_covariance

    def identity_innovation(
        normalized_observation_deviations,
        chol_observation_covariance,
        model_inputs,
    ):
        del normalized_observation_deviations, model_inputs
        return chol_observation_covariance

    def bad_cross(cross_covariance, model_inputs):
        del cross_covariance, model_inputs
        return jnp.ones((1, 1))

    bad_cross_resolved = resolve_enkf_localization(
        EnKFLocalizationFunctions(modify_cross_covariance=bad_cross),
        state_dim=3,
        observation_dim=2,
    )
    assert bad_cross_resolved.modify_cross_covariance is not None
    with pytest.raises(ValueError, match="modify_cross_covariance output.*shape"):
        bad_cross_resolved.modify_cross_covariance(jnp.ones((3, 2)), None)

    def bad_innovation(
        normalized_observation_deviations,
        chol_observation_covariance,
        model_inputs,
    ):
        del normalized_observation_deviations, chol_observation_covariance, model_inputs
        return jnp.full((2, 2), jnp.nan)

    bad_innovation_resolved = resolve_enkf_localization(
        EnKFLocalizationFunctions(
            construct_chol_innovation_covariance=bad_innovation,
            modify_predicted_observation_covariance=identity_prediction,
        ),
        state_dim=3,
        observation_dim=2,
    )
    assert bad_innovation_resolved.construct_chol_innovation_covariance is not None
    with pytest.raises(ValueError, match="only finite"):
        bad_innovation_resolved.construct_chol_innovation_covariance(
            jnp.ones((2, 4)),
            jnp.eye(2),
            None,
        )

    def bad_prediction(predicted_observation_covariance, model_inputs):
        del predicted_observation_covariance, model_inputs
        return jnp.ones((2, 1))

    bad_prediction_resolved = resolve_enkf_localization(
        EnKFLocalizationFunctions(
            construct_chol_innovation_covariance=identity_innovation,
            modify_predicted_observation_covariance=bad_prediction,
        ),
        state_dim=3,
        observation_dim=2,
    )
    assert bad_prediction_resolved.modify_predicted_observation_covariance is not None
    with pytest.raises(
        ValueError,
        match="modify_predicted_observation_covariance output.*shape",
    ):
        bad_prediction_resolved.modify_predicted_observation_covariance(
            jnp.eye(2),
            None,
        )
