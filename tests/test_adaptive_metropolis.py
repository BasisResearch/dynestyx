"""Joint proposal, adaptation, and stochastic-density retention regressions."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer.initialization import init_to_value

import dynestyx.inference.integrations.blackjax.mcmc as blackjax_mcmc
from dynestyx.inference.configs.mcmc import AdaptiveMetropolisConfig, AdaptiveMWGConfig
from dynestyx.inference.integrations.blackjax.adaptive_metropolis import (
    adaptive_metropolis,
    proposal_covariance,
)
from dynestyx.inference.mcmc import MCMCInference


def _algorithm(logdensity_fn, scale=0.02, warmup=2):
    return adaptive_metropolis(
        logdensity_fn,
        jnp.asarray(scale),
        target_acceptance_rate=0.234,
        adaptation_rate=0.6,
        num_warmup=warmup,
    )


def test_config_defaults_and_mwg_rename():
    config = AdaptiveMetropolisConfig(num_samples=4, num_warmup=2, num_chains=1)
    assert config.mcmc_source == "blackjax"
    assert config.initial_proposal_scale == 0.02
    assert config.target_acceptance_rate == 0.234
    assert config.adaptation_rate == 0.6
    assert not hasattr(config, "max_adaptation")
    mwg = AdaptiveMWGConfig(num_samples=4, num_warmup=2, num_chains=1)
    assert mwg.initial_proposal_scale == 1.0
    assert mwg.target_acceptance_rate == 0.44
    assert mwg.adaptation_rate == 0.5
    assert mwg.max_adaptation == 0.01


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mcmc_source": "numpyro"}, "only supports"),
        *[
            ({"target_acceptance_rate": x}, "between 0 and 1")
            for x in [0, 1, np.nan, np.inf]
        ],
        *[
            ({"adaptation_rate": x}, r"finite and in \(0.5, 1\]")
            for x in [0, 0.5, 1.1, np.nan, np.inf]
        ],
        ({"initial_proposal_scale": np.array([[1.0]])}, "scalar or 1D"),
        ({"initial_proposal_scale": np.array([])}, "finite values"),
        ({"initial_proposal_scale": np.array([1.0, np.nan])}, "finite values"),
        ({"initial_proposal_scale": np.inf}, "finite values"),
        ({"initial_proposal_scale": 0}, "positive"),
        ({"initial_proposal_scale": np.array([1.0, -0.1])}, "positive"),
    ],
)
def test_invalid_config(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AdaptiveMetropolisConfig(num_samples=4, num_warmup=2, num_chains=1, **kwargs)


def test_initialization_and_joint_proposal():
    scales = jnp.array([0.2, 0.3, 0.4])
    algorithm = _algorithm(lambda x: jnp.array(0.0), scales, warmup=0)
    initial = algorithm.init(jnp.zeros(3))
    np.testing.assert_array_equal(initial.mean, initial.position)
    np.testing.assert_allclose(initial.covariance, jnp.diag(scales**2))
    np.testing.assert_allclose(jnp.exp(initial.log_multiplier), 2.38**2 / 3)

    key = jr.PRNGKey(2)
    # RMH splits for proposal/acceptance; its outer kernel makes one proposal.
    proposal_key, _ = jr.split(key)
    expected = jnp.linalg.cholesky(proposal_covariance(initial)) @ jr.normal(
        proposal_key, (3,)
    )
    next_state, info = algorithm.step(key, initial)
    np.testing.assert_allclose(next_state.position, expected, rtol=1e-6)
    assert bool(info.is_accepted)
    assert float(info.acceptance_rate) == 1.0
    assert np.count_nonzero(next_state.position) == 3


def test_exact_adaptation_uses_probability_and_previous_mean():
    algorithm = _algorithm(lambda x: -0.5 * jnp.sum(x**2), scale=0.5, warmup=4)
    state = algorithm.init(jnp.array([0.2, -0.1]))
    # Use nonzero iteration and distinct mean to expose old/new-mean mistakes.
    state = state._replace(mean=jnp.array([0.5, 0.7]), n_iter=jnp.array(2))
    next_state, info = algorithm.step(jr.PRNGKey(0), state)
    gain = 3.0**-0.6
    delta = next_state.position - state.mean
    assert 0.0 < float(info.acceptance_rate) < 1.0
    np.testing.assert_allclose(next_state.mean, state.mean + gain * delta, rtol=1e-6)
    np.testing.assert_allclose(
        next_state.covariance,
        state.covariance + gain * (jnp.outer(delta, delta) - state.covariance),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        next_state.log_multiplier,
        state.log_multiplier + gain * (info.acceptance_rate - 0.234),
        rtol=1e-6,
    )
    assert int(next_state.n_iter) == 3


@pytest.mark.parametrize("warmup", [0, 1, 3])
def test_adaptation_freezes_at_warmup_boundary(warmup):
    algorithm = _algorithm(lambda x: -0.5 * jnp.sum(x**2), warmup=warmup)
    state = algorithm.init(jnp.zeros(2))
    initial = state
    for key in jr.split(jr.PRNGKey(4), warmup):
        state, _ = algorithm.step(key, state)
    if warmup:
        assert not np.array_equal(state.covariance, initial.covariance)
    frozen = state
    for key in jr.split(jr.PRNGKey(5), 3):
        state, _ = algorithm.step(key, state)
        for field in ["mean", "covariance", "log_multiplier"]:
            np.testing.assert_array_equal(getattr(state, field), getattr(frozen, field))


def test_density_evaluated_once_and_retained_on_rejection():
    evaluations = []

    def density(position):
        evaluations.append(position)
        return jnp.where(jnp.all(position == 0), -7.0, -jnp.inf)

    algorithm = _algorithm(density)
    state = algorithm.init(jnp.zeros(2))
    assert len(evaluations) == 1
    for i, key in enumerate(jr.split(jr.PRNGKey(3), 4)):
        state, info = algorithm.step(key, state)
        assert len(evaluations) == i + 2
        assert not bool(info.is_accepted)
        np.testing.assert_array_equal(state.position, jnp.zeros(2))
        assert float(state.logdensity) == -7.0


def test_accepted_density_becomes_cached_current_estimate():
    initial_algorithm = _algorithm(lambda x: jnp.array(-10.0), warmup=0)
    state = initial_algorithm.init(jnp.zeros(2))
    accepting_algorithm = _algorithm(lambda x: jnp.array(-9.0), warmup=0)
    state, info = accepting_algorithm.step(jr.PRNGKey(0), state)
    assert bool(info.is_accepted)
    assert float(state.logdensity) == -9.0
    rejecting_algorithm = _algorithm(lambda x: jnp.array(-jnp.inf), warmup=0)
    rejected, info = rejecting_algorithm.step(jr.PRNGKey(1), state)
    assert not bool(info.is_accepted)
    np.testing.assert_array_equal(rejected.position, state.position)
    assert float(rejected.logdensity) == -9.0


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_proposal_is_rejected(invalid):
    algorithm = _algorithm(lambda x: jnp.where(jnp.all(x == 0), 0.0, invalid), warmup=0)
    initial = algorithm.init(jnp.zeros(2))
    state, info = jax.jit(algorithm.step)(jr.PRNGKey(7), initial)
    assert not bool(info.is_accepted)
    assert float(info.acceptance_rate) == 0.0
    np.testing.assert_array_equal(state.position, initial.position)
    assert float(state.logdensity) == 0.0


def test_finite_proposal_escapes_negative_infinity_and_undefined_ratio_rejects():
    algorithm = _algorithm(lambda x: jnp.array(-jnp.inf), warmup=0)
    initial = algorithm.init(jnp.zeros(2))
    assert float(initial.logdensity) == -np.inf
    state, info = algorithm.step(jr.PRNGKey(8), initial)
    assert not bool(info.is_accepted)
    assert float(info.acceptance_rate) == 0.0
    finite_algorithm = _algorithm(lambda x: jnp.array(-10.0), warmup=0)
    state, info = finite_algorithm.step(jr.PRNGKey(9), state)
    assert bool(info.is_accepted)
    assert float(info.acceptance_rate) == 1.0
    assert float(state.logdensity) == -10.0


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("reject", [False, True])
def test_singular_first_update_remains_finite(dtype, reject):
    with jax.enable_x64(True):
        density = (
            (lambda x: jnp.where(jnp.all(x == 0), 0.0, -jnp.inf))
            if reject
            else (lambda x: jnp.asarray(0.0, dtype=dtype))
        )
        algorithm = _algorithm(density)
        state = algorithm.init(jnp.zeros(10, dtype=dtype))
        state, _ = algorithm.step(jr.PRNGKey(10), state)
        covariance = proposal_covariance(state)
        np.testing.assert_array_equal(covariance, covariance.T)
        assert bool(jnp.all(jnp.isfinite(jnp.linalg.cholesky(covariance))))
        state, _ = jax.jit(algorithm.step)(jr.PRNGKey(11), state)
        assert state.position.dtype == dtype
        assert all(bool(jnp.all(jnp.isfinite(x))) for x in state)


@pytest.mark.parametrize("num_chains", [1, 2])
@pytest.mark.parametrize("num_warmup", [0, 2])
def test_structured_constrained_multichain_run(num_chains, num_warmup):
    def model(obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None):
        del obs_times, obs_values, ctrl_times, ctrl_values
        numpyro.sample("a", dist.Normal(0.0, 1.0))
        numpyro.sample("b", dist.LogNormal(jnp.zeros(2), 1.0).to_event(1))

    inference = MCMCInference(
        AdaptiveMetropolisConfig(
            num_samples=4,
            num_warmup=num_warmup,
            num_chains=num_chains,
            initial_proposal_scale=jnp.array([0.2, 0.3, 0.4]),
            init_strategy=init_to_value(values={"a": 0.0, "b": jnp.ones(2)}),
        ),
        model,
    )
    samples = inference.run(jr.PRNGKey(12), jnp.zeros(1), jnp.zeros(1))
    diagnostics = inference.get_diagnostics()
    assert samples["a"].shape == (num_chains, 4)
    assert samples["b"].shape == (num_chains, 4, 2)
    assert bool(jnp.all(samples["b"] > 0))
    assert diagnostics["mean_acceptance_rate"].shape == (num_chains,)
    assert diagnostics["final_global_scale"].shape == (num_chains,)
    assert diagnostics["final_proposal_covariance"].shape == (num_chains, 3, 3)
    assert bool(jnp.all(jnp.isfinite(diagnostics["final_proposal_covariance"])))
    repeated = inference.run(jr.PRNGKey(12), jnp.zeros(1), jnp.zeros(1))
    for name in samples:
        np.testing.assert_array_equal(repeated[name], samples[name])
    for name in diagnostics:
        np.testing.assert_array_equal(
            inference.get_diagnostics()[name], diagnostics[name]
        )


def test_initial_scale_dimension_checked_by_integration():
    def model(obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None):
        numpyro.sample("x", dist.Normal(jnp.zeros(3), 1.0).to_event(1))

    inference = MCMCInference(
        AdaptiveMetropolisConfig(
            num_samples=1,
            num_warmup=0,
            num_chains=1,
            initial_proposal_scale=jnp.array([0.1, 0.2]),
        ),
        model,
    )
    with pytest.raises(ValueError, match=r"expected \(3,\), got \(2,\)"):
        inference.run(jr.PRNGKey(0), jnp.zeros(1), jnp.zeros(1))


def test_independent_density_keys_and_sampling_acceptance(monkeypatch):
    evaluations = []

    def fake_init_model(*args, **kwargs):
        num_chains = kwargs["rng_key"].shape[0]

        def potential_gen(*args, **kwargs):
            def potential(position, key):
                is_initial = jnp.all(position["x"] == 0)
                jax.debug.callback(
                    lambda k, initial: evaluations.append(
                        (np.asarray(k), bool(initial))
                    ),
                    key,
                    is_initial,
                )
                # All proposals reject, exposing accidental current-density refresh.
                return jnp.where(is_initial, jr.normal(key), jnp.inf)

            return potential

        return (
            SimpleNamespace(z={"x": jnp.zeros((num_chains, 2))}),
            potential_gen,
            lambda *args, **kwargs: lambda position: position,
        )

    monkeypatch.setattr(blackjax_mcmc, "init_model", fake_init_model)
    inference = MCMCInference(
        AdaptiveMetropolisConfig(num_samples=3, num_warmup=1, num_chains=2),
        lambda *args: None,
    )
    samples = inference.run(jr.PRNGKey(13), jnp.zeros(1), jnp.zeros(1))
    jax.block_until_ready(samples)
    jax.effects_barrier()
    assert len(evaluations) == 2 * (1 + 1 + 3)
    assert sum(initial for _, initial in evaluations) == 2
    keys = [tuple(key.tolist()) for key, _ in evaluations]
    assert len(set(keys)) == len(keys)
    np.testing.assert_array_equal(samples["x"], np.zeros((2, 3, 2)))
    np.testing.assert_array_equal(
        inference.get_diagnostics()["mean_acceptance_rate"], [0.0, 0.0]
    )


def test_joint_diagnostics_exclude_warmup(monkeypatch):
    original = blackjax_mcmc.adaptive_metropolis

    def algorithm_with_known_indicators(*args, **kwargs):
        algorithm = original(*args, **kwargs)

        def step(key, state):
            next_state, info = algorithm.step(key, state)
            return next_state, info._replace(
                is_accepted=state.n_iter < kwargs["num_warmup"]
            )

        return algorithm._replace(step=step)

    monkeypatch.setattr(
        blackjax_mcmc, "adaptive_metropolis", algorithm_with_known_indicators
    )

    def model(obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None):
        numpyro.sample("x", dist.Normal(0.0, 1.0))

    inference = MCMCInference(
        AdaptiveMetropolisConfig(num_samples=3, num_warmup=2, num_chains=1), model
    )
    inference.run(jr.PRNGKey(14), jnp.zeros(1), jnp.zeros(1))
    np.testing.assert_array_equal(
        inference.get_diagnostics()["mean_acceptance_rate"], [0.0]
    )


def test_correlated_gaussian_posterior_and_covariance():
    target_covariance = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    precision = jnp.linalg.inv(target_covariance)
    algorithm = _algorithm(lambda x: -0.5 * x @ precision @ x, warmup=2000)
    initial = algorithm.init(jnp.zeros(2))

    @jax.jit
    def run(key):
        def step(state, key):
            state, _ = algorithm.step(key, state)
            return state, state.position

        return jax.lax.scan(step, initial, jr.split(key, 8000))

    final, positions = run(jr.PRNGKey(42))
    samples = np.asarray(positions[2000:])
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.15)
    np.testing.assert_allclose(np.cov(samples.T), target_covariance, atol=0.25)
    assert abs(np.corrcoef(samples.T)[0, 1] - 0.8) < 0.1
    learned = np.asarray(final.covariance)
    assert learned[0, 1] > 0.0
    assert abs(learned[0, 1] / np.sqrt(learned[0, 0] * learned[1, 1]) - 0.8) < 0.2
