"""LETKF integration: numerical reference, validation, and public handler paths."""

import dataclasses
from contextlib import nullcontext

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from cuthbert import filter as cuthbert_filter
from cuthbert.ensemble_kalman import local_ensemble_transform_kalman_filter as letkf
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.evaluation import Evaluation, ObservationScoringConfig
from dynestyx.evaluation.scoring import GaussianLogProbScore
from dynestyx.inference.filters import Filter, LETKFConfig
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    build_cuthbert_filter,
    compute_cuthbert_filter,
)
from dynestyx.inference.letkf_localization import validate_letkf_localization
from dynestyx.inference.observation_predictions import PredictedObservationOutputs


def _problem(alpha=0.8, observation_model=None):
    return dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.array([0.1, -0.2]), jnp.eye(2)),
        state_evolution=dsx.LinearGaussianStateEvolution(
            A=alpha * jnp.eye(2), cov=0.05 * jnp.eye(2)
        ),
        observation_model=(
            dsx.LinearGaussianObservation(
                H=jnp.eye(2), R=jnp.diag(jnp.array([0.2, 0.4]))
            )
            if observation_model is None
            else observation_model
        ),
    )


def _config(**kwargs):
    return LETKFConfig(
        observation_indices=jnp.array([[0, 1], [0, 1]]),
        localization_weights=jnp.ones((2, 2)),
        n_particles=8,
        **kwargs,
    )


TIMES = jnp.arange(3.0)
VALUES = jnp.array([[0.4, -0.3], [0.2, 0.5], [-0.1, 0.2]])


def _run(config=None, values=VALUES, model=None):
    return compute_cuthbert_filter(
        _problem() if model is None else model,
        _config() if config is None else config,
        jr.PRNGKey(7),
        obs_times=TIMES,
        obs_values=values,
    )


def test_matches_direct_cuthbert_and_linear_gaussian_analysis():
    model, config = _problem(), _config(inflation_delta=0.1)
    ll, states = _run(config, model=model)

    def get_dynamics(mi):
        return lambda x, key: jax.lax.cond(
            mi.is_first_step,
            lambda: x,
            lambda: model.state_evolution(x, mi.u_prev, mi.time_prev, mi.time).sample(
                key
            ),
        )

    reference = letkf.build_filter(
        init_sample=lambda key: model.initial_condition.sample(key),
        get_dynamics=get_dynamics,
        get_observations=lambda mi: (
            lambda x: x,
            jnp.sqrt(jnp.array([0.2, 0.4])),
            mi.y,
        ),
        n_particles=8,
        get_localization=lambda mi: (
            config.observation_indices,
            config.localization_weights,
        ),
        inflation=0.1,
        store_predicted_ensemble=True,
    )
    init_key, filter_key = jr.split(jr.PRNGKey(7))
    expected = cuthbert_filter(
        reference,
        states.model_inputs,
        reference.init_prepare(key=init_key),
        key=filter_key,
        parallel=False,
    )
    assert jnp.allclose(states.ensemble, expected.ensemble[1:], atol=2e-6)
    assert jnp.allclose(ll, expected.log_normalizing_constant[-1])

    forecast = states.predicted_ensemble[0]
    mean, cov = forecast.mean(0), jnp.cov(forecast.T)
    noise = jnp.diag(jnp.array([0.2, 0.4]))
    gain = jnp.linalg.solve(cov + noise, cov).T
    expected_mean = mean + gain @ (VALUES[0] - mean)
    expected_cov = cov - gain @ cov
    assert jnp.allclose(states.ensemble[0].mean(0), expected_mean, atol=2e-6)
    assert jnp.allclose(jnp.cov(states.ensemble[0].T), expected_cov, atol=2e-6)
    expected_ll = dist.MultivariateNormal(mean, cov + noise).log_prob(VALUES[0])
    assert jnp.allclose(states.log_normalizing_constant[0], expected_ll, atol=2e-6)


def test_local_weights_padding_missingness_and_global_score():
    config = LETKFConfig(
        observation_indices=jnp.array([[0, -1], [1, -1]]),
        localization_weights=jnp.array([[0.3, jnp.nan], [0.0, -2.0]]),
        n_particles=8,
    )
    values = VALUES.at[0, 1].set(jnp.nan).at[1].set(jnp.nan)
    _, states = _run(config, values)
    forecast = states.predicted_ensemble[0]
    var = jnp.var(forecast[:, 0], ddof=1)
    gain = var / (var + 0.2 / 0.3)
    expected_mean = forecast[:, 0].mean() + gain * (
        values[0, 0] - forecast[:, 0].mean()
    )
    assert jnp.allclose(states.ensemble[0, :, 0].mean(), expected_mean, atol=2e-6)
    assert jnp.allclose(
        jnp.var(states.ensemble[0, :, 0], ddof=1), var * (1 - gain), atol=2e-6
    )
    assert jnp.array_equal(states.ensemble[0, :, 1], forecast[:, 1])
    assert jnp.array_equal(states.ensemble[1], states.predicted_ensemble[1])
    assert states.log_normalizing_constant[1] == states.log_normalizing_constant[0]

    _, original = _run(config)
    _, changed = _run(config, VALUES.at[0, 1].add(3))
    assert jnp.array_equal(original.ensemble[0], changed.ensemble[0])
    assert not jnp.isclose(
        original.log_normalizing_constant[0], changed.log_normalizing_constant[0]
    )
    _, reweighted = _run(
        dataclasses.replace(
            config, localization_weights=jnp.array([[0.8, 0.0], [0.0, 0.0]])
        )
    )
    assert jnp.allclose(
        original.log_normalizing_constant[0], reweighted.log_normalizing_constant[0]
    )
    assert not jnp.allclose(original.ensemble[0], reweighted.ensemble[0])

    empty = LETKFConfig(
        observation_indices=jnp.empty((2, 0), dtype=int),
        localization_weights=jnp.empty((2, 0)),
        n_particles=8,
    )
    ll, states = _run(empty)
    assert jnp.array_equal(states.ensemble, states.predicted_ensemble)
    assert jnp.isfinite(ll)
    ll, states = _run(values=jnp.full_like(VALUES, jnp.nan))
    assert ll == 0
    assert jnp.array_equal(states.ensemble, states.predicted_ensemble)


def test_time_varying_callback_and_jit_gradients():
    indices = jnp.array([[0, 1], [0, 1]])

    def objective(alpha, weight):
        config = LETKFConfig(
            n_particles=8,
            get_localization=lambda mi: (
                indices,
                jnp.ones((2, 2)) * weight * (mi.time > 0),
            ),
        )
        ll, states = _run(config, model=_problem(alpha))
        return ll + jnp.sum(states.ensemble[-1])

    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(0.8, 0.5)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(jnp.array(gradients)))
    assert abs(gradients[1]) > 1e-5

    config = LETKFConfig(
        n_particles=8,
        get_localization=lambda mi: (indices, jnp.ones((2, 2)) * (mi.time > 0)),
    )
    _, states = _run(config)
    assert jnp.array_equal(states.ensemble[0], states.predicted_ensemble[0])
    assert not jnp.allclose(states.ensemble[1], states.predicted_ensemble[1])


@pytest.mark.parametrize(
    "indices,weights,message",
    [
        (jnp.array([[0.0], [1.0]]), jnp.ones((2, 1)), "integers"),
        (jnp.array([[0]]), jnp.ones((1, 1)), "shape"),
        (jnp.array([[0], [1]]), jnp.ones((2, 2)), "match"),
        (jnp.array([[0], [2]]), jnp.ones((2, 1)), "range"),
        (jnp.array([[-2], [1]]), jnp.ones((2, 1)), "range"),
        (jnp.array([[0, 0], [1, -1]]), jnp.ones((2, 2)), "unique"),
        (jnp.array([[0], [1]]), jnp.array([[-1.0], [1.0]]), "nonnegative"),
        (jnp.array([[0], [1]]), jnp.array([[jnp.nan], [1.0]]), "finite"),
    ],
)
def test_neighborhood_validation(indices, weights, message):
    with pytest.raises(ValueError, match=message):
        build_cuthbert_filter(
            _problem(),
            LETKFConfig(observation_indices=indices, localization_weights=weights),
            jr.PRNGKey(0),
            want_parallel=False,
        )


@pytest.mark.parametrize(
    "bad_indices,bad_weights,message",
    [
        (jnp.array([[0], [3]]), jnp.ones((2, 1)), "range"),
        (jnp.array([[0], [1]]), jnp.array([[1.0], [-1.0]]), "nonnegative"),
    ],
)
def test_callback_validation_under_jit(bad_indices, bad_weights, message):
    def run(indices, weights):
        config = LETKFConfig(
            get_localization=lambda mi: (indices, weights), n_particles=4
        )
        return _run(config)[0]

    with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match=message):
        jax.jit(run)(bad_indices, bad_weights).block_until_ready()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"observation_indices": jnp.zeros((2, 1), dtype=int)},
        {"get_localization": 3},
        {"get_localization": lambda mi: None, "observation_indices": [[0], [1]]},
        {
            "observation_indices": [[0], [1]],
            "localization_weights": [[1], [1]],
            "n_particles": 1,
        },
        {
            "observation_indices": [[0], [1]],
            "localization_weights": [[1], [1]],
            "filter_source": "cd_dynamax",
        },
        {
            "observation_indices": [[0], [1]],
            "localization_weights": [[1], [1]],
            "extra_filter_kwargs": {"perturbed_obs": True},
        },
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LETKFConfig(**kwargs)


@pytest.mark.parametrize("kind", ["linear", "gaussian", "normal", "independent", "mvn"])
def test_diagonal_observation_adapters(kind):
    variances = jnp.array([0.2, 0.4])
    models = {
        "linear": dsx.LinearGaussianObservation(H=jnp.eye(2), R=jnp.diag(variances)),
        "gaussian": dsx.GaussianObservation(
            h=lambda x, u, t: x, R=dsx.DiagonalCovariance(variance=variances)
        ),
        "normal": lambda x, u, t: dist.Normal(x, jnp.sqrt(variances)),
        "independent": lambda x, u, t: dist.Normal(x, jnp.sqrt(variances)).to_event(1),
        "mvn": lambda x, u, t: dist.MultivariateNormal(x, jnp.diag(variances)),
    }
    ll, states = _run(model=_problem(observation_model=models[kind]))
    reference_ll, reference = _run()
    assert jnp.allclose(ll, reference_ll, atol=2e-6)
    assert jnp.allclose(states.ensemble, reference.ensemble, atol=2e-6)


def test_noise_rejection_eager_and_jit():
    correlated = dsx.LinearGaussianObservation(
        H=jnp.eye(2), R=jnp.array([[0.2, 0.1], [0.1, 0.4]])
    )
    with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match="diagonal"):
        _run(model=_problem(observation_model=correlated))

    def run(offdiag):
        noise = jnp.array([[0.2, offdiag], [offdiag, 0.4]])
        return _run(
            model=_problem(
                observation_model=dsx.LinearGaussianObservation(H=jnp.eye(2), R=noise)
            )
        )[0]

    with pytest.raises((ValueError, jax.errors.JaxRuntimeError), match="diagonal"):
        jax.jit(run)(0.1).block_until_ready()

    heteroscedastic = lambda x, u, t: dist.Normal(x, jnp.exp(x))
    with pytest.raises(ValueError, match="state-independent"):
        _run(model=_problem(observation_model=heteroscedastic))


@pytest.mark.parametrize("plated", [False, True])
def test_public_condition_predictions_and_scores(plated):
    config = _config()

    with (
        Evaluation(
            observation_scoring_config=ObservationScoringConfig(
                rules=(GaussianLogProbScore(),)
            )
        ),
        Filter(filter_config=config),
        dsx.plate("members", 2) if plated else nullcontext(),
    ):
        result = dsx.condition(
            "f",
            _problem(),
            obs_times=TIMES,
            obs_values=jnp.stack([VALUES, VALUES]) if plated else VALUES,
        )
    loglik = result.marginal_loglik
    assert loglik is not None
    assert jnp.all(jnp.isfinite(loglik))
    predictions = result.predicted_observations
    assert isinstance(predictions, PredictedObservationOutputs)
    assert predictions.mean is not None
    assert predictions.ensemble is not None
    assert predictions.obs_cov is not None
    prefix = (2,) if plated else ()
    assert predictions.mean.shape == (*prefix, 3, 2)
    assert predictions.ensemble.shape == (*prefix, 3, 8, 2)
    evaluation = result.evaluation_result
    assert evaluation is not None
    assert result.dists is not None
    expected_scores = dist.MultivariateNormal(
        predictions.mean, predictions.obs_cov
    ).log_prob(jnp.stack([VALUES, VALUES]) if plated else VALUES)
    score = next(iter(evaluation.observation_scores.values()))
    assert jnp.allclose(score.squeeze(-1), expected_scores, atol=2e-6)

    if not plated:

        def model():
            with Filter(filter_config=config):
                return dsx.sample("f", _problem(), obs_times=TIMES, obs_values=VALUES)

        traced = trace(seed(model, jr.PRNGKey(1))).get_trace()
        assert "f_marginal_loglik" in traced
        assert "f_predicted_observations_ensemble" in traced
        _, disabled = _run(
            dataclasses.replace(config, include_predicted_observations=False)
        )
        assert disabled.predicted_ensemble is None


def test_validation_helper_compiles_valid_empty_neighborhoods():
    validate = jax.jit(
        lambda i, w: validate_letkf_localization(i, w, state_dim=2, observation_dim=2)
    )
    indices, weights = validate(jnp.empty((2, 0), dtype=int), jnp.empty((2, 0)))
    assert indices.shape == weights.shape == (2, 0)


@pytest.mark.parametrize("variance", [0.0, -0.1, jnp.inf])
def test_invalid_observed_noise(variance):
    def observation(x, u, t):
        return dist.Normal(x, jnp.array([variance, 0.4]))

    with pytest.raises(
        (ValueError, jax.errors.JaxRuntimeError), match="positive|invalid scale"
    ):
        _run(model=_problem(observation_model=observation))


def test_scalar_noise():
    scalar = dsx.GaussianObservation(
        h=lambda x, u, t: x, R=dsx.ScalarCovariance(variance=0.2)
    )
    dense = dsx.LinearGaussianObservation(H=jnp.eye(2), R=0.2 * jnp.eye(2))
    scalar_ll, scalar_states = _run(model=_problem(observation_model=scalar))
    dense_ll, dense_states = _run(model=_problem(observation_model=dense))
    assert jnp.allclose(scalar_ll, dense_ll, atol=2e-6)
    assert jnp.allclose(scalar_states.ensemble, dense_states.ensemble, atol=2e-6)


def test_custom_noise_independence_under_jit():
    def run(alpha):
        observation = lambda x, u, t: dist.Normal(x, jnp.exp(alpha * x))
        return _run(model=_problem(observation_model=observation))[0]

    with pytest.raises(jax.errors.JaxRuntimeError, match="state-independent"):
        jax.jit(run)(0.1).block_until_ready()
