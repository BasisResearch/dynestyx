"""Behavioral coverage for the joint sampler; MWG regressions live separately."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer.initialization import init_to_value

from dynestyx.inference.configs.mcmc import AdaptiveMetropolisConfig
from dynestyx.inference.integrations.blackjax.adaptive_metropolis import (
    adaptive_metropolis,
    proposal_covariance,
)
from dynestyx.inference.mcmc import MCMCInference
from tests.test_utils import assert_finite


def _algorithm(logdensity_fn, scale=0.02, warmup=0):
    return adaptive_metropolis(
        logdensity_fn,
        jnp.asarray(scale),
        target_acceptance_rate=0.234,
        adaptation_rate=0.6,
        num_warmup=warmup,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mcmc_source": "numpyro"},
        {"initial_proposal_scale": 0},
        {"initial_proposal_scale": np.nan},
        {"target_acceptance_rate": 1},
        {"adaptation_rate": 0.5},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        AdaptiveMetropolisConfig(num_samples=4, num_warmup=2, num_chains=1, **kwargs)


@pytest.mark.parametrize("warmup", [0, 2])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_joint_moves_and_warmup_freezing(warmup, dtype):
    with jax.enable_x64(True):
        algorithm = _algorithm(lambda x: jnp.asarray(0.0), [0.2, 0.3, 0.4], warmup)
        step = jax.jit(algorithm.step)
        state = algorithm.init(jnp.zeros(3, dtype=dtype))
        for i, key in enumerate(jr.split(jr.PRNGKey(4), warmup + 2)):
            previous = state
            state, info = step(key, state)
            assert bool(info.is_accepted)
            assert np.all(state.position != previous.position)
            assert state.position.dtype == dtype
            assert_finite(state.position, (3,))
            # The first covariance update is singular; subsequent proposals must work.
            assert_finite(jnp.linalg.cholesky(proposal_covariance(state)), (3, 3))
            if i < warmup:
                assert not np.array_equal(state.covariance, previous.covariance)
                assert state.log_multiplier > previous.log_multiplier
            else:
                for field in ("mean", "covariance", "log_multiplier"):
                    np.testing.assert_array_equal(
                        getattr(state, field), getattr(previous, field)
                    )


def test_density_is_evaluated_once_per_proposal_and_cached():
    evaluations = []

    def density(position):
        evaluations.append(position)
        return jnp.asarray([-10.0, -9.0, -jnp.inf][len(evaluations) - 1])

    algorithm = _algorithm(density)
    initial = algorithm.init(jnp.zeros(2))
    assert len(evaluations) == 1
    accepted, info = algorithm.step(jr.PRNGKey(0), initial)
    assert bool(info.is_accepted)
    assert float(accepted.logdensity) == -9.0
    assert len(evaluations) == 2
    rejected, info = algorithm.step(jr.PRNGKey(1), accepted)
    assert not bool(info.is_accepted)
    np.testing.assert_array_equal(rejected.position, accepted.position)
    assert float(rejected.logdensity) == -9.0
    assert len(evaluations) == 3


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_nonfinite_densities_reject_but_finite_proposals_can_escape(invalid):
    algorithm = _algorithm(lambda x: jnp.where(jnp.all(x == 0), 0.0, invalid))
    initial = algorithm.init(jnp.zeros(2))
    state, info = jax.jit(algorithm.step)(jr.PRNGKey(7), initial)
    assert not bool(info.is_accepted)
    assert float(info.acceptance_rate) == 0.0
    np.testing.assert_array_equal(state.position, initial.position)
    assert float(state.logdensity) == 0.0

    invalid_state = algorithm.init(jnp.ones(2))
    assert float(invalid_state.logdensity) == -np.inf
    finite_algorithm = _algorithm(lambda x: jnp.asarray(0.0))
    state, info = finite_algorithm.step(jr.PRNGKey(8), invalid_state)
    assert bool(info.is_accepted)
    assert float(state.logdensity) == 0.0


@pytest.mark.parametrize("num_chains", [1, 2])
def test_structured_constrained_samples_and_diagnostics(num_chains):
    def model(obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None):
        numpyro.sample("a", dist.Normal(0.0, 1.0))
        numpyro.sample("b", dist.LogNormal(jnp.zeros(2), 1.0).to_event(1))

    inference = MCMCInference(
        AdaptiveMetropolisConfig(
            num_samples=4,
            num_warmup=2,
            num_chains=num_chains,
            initial_proposal_scale=jnp.array([0.2, 0.3, 0.4]),
            init_strategy=init_to_value(values={"a": 0.0, "b": jnp.ones(2)}),
        ),
        model,
    )
    samples = inference.run(jr.PRNGKey(12), jnp.zeros(1), jnp.zeros(1))
    diagnostics = inference.get_diagnostics()
    assert_finite(samples["a"], (num_chains, 4))
    assert_finite(samples["b"], (num_chains, 4, 2))
    assert np.all(samples["b"] > 0)
    assert_finite(diagnostics["mean_acceptance_rate"], (num_chains,))
    assert_finite(diagnostics["final_global_scale"], (num_chains,))
    assert_finite(diagnostics["final_proposal_covariance"], (num_chains, 3, 3))
    repeated = inference.run(jr.PRNGKey(12), jnp.zeros(1), jnp.zeros(1))
    for name in samples:
        np.testing.assert_array_equal(repeated[name], samples[name])


def test_correlated_gaussian_posterior():
    target_covariance = jnp.array([[1.0, 0.8], [0.8, 1.0]])
    precision = jnp.linalg.inv(target_covariance)
    algorithm = _algorithm(lambda x: -0.5 * x @ precision @ x, warmup=2000)

    @jax.jit
    def run(key):
        def step(state, key):
            state, _ = algorithm.step(key, state)
            return state, state.position

        return jax.lax.scan(step, algorithm.init(jnp.zeros(2)), jr.split(key, 8000))

    final, positions = run(jr.PRNGKey(42))
    samples = np.asarray(positions[2000:])
    np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.15)
    np.testing.assert_allclose(np.cov(samples.T), target_covariance, atol=0.25)
    assert final.covariance[0, 1] > 0
