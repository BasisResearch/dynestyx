import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer.initialization import init_to_value

import dynestyx.inference.integrations.blackjax.mcmc as blackjax_mcmc_module
import dynestyx.inference.mcmc as mcmc_module
from dynestyx.inference.configs.mcmc import (
    AdaptiveMetropolisConfig,
    NUTSConfig,
)
from dynestyx.inference.integrations.blackjax.adaptive_metropolis import (
    adapt_proposal_scale,
    adaptive_metropolis,
    resolve_proposal_scale,
)
from dynestyx.inference.mcmc import MCMCInference


def _normal_model(obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None):
    del obs_times, obs_values, ctrl_times, ctrl_values
    numpyro.sample("x", dist.Normal(0.0, 1.0))


def test_adaptation_rule():
    updated = adapt_proposal_scale(
        jnp.ones(2),
        jnp.array([0.2, 0.8]),
        jnp.array(4.0),
        target_acceptance_rate=0.44,
        adaptation_rate=0.5,
        max_adaptation=0.1,
    )
    np.testing.assert_allclose(updated, jnp.exp(jnp.array([-0.1, 0.1])))


def test_kernel_stops_adapting_after_warmup():
    algorithm = adaptive_metropolis(
        lambda x: -0.5 * jnp.sum(x**2),
        jnp.ones(2),
        target_acceptance_rate=0.44,
        adaptation_rate=0.5,
        max_adaptation=0.1,
        num_warmup=1,
    )
    initial_state = algorithm.init(jnp.zeros(2))
    warm_state, _ = algorithm.step(jr.PRNGKey(0), initial_state)
    sample_state, _ = algorithm.step(jr.PRNGKey(1), warm_state)

    assert not jnp.array_equal(warm_state.proposal_scale, jnp.ones(2))
    np.testing.assert_array_equal(
        sample_state.proposal_scale,
        warm_state.proposal_scale,
    )


def test_proposal_scale_matches_flattened_latent_dimension():
    with pytest.raises(ValueError, match=r"expected \(3,\), got \(2,\)"):
        resolve_proposal_scale(jnp.array([0.1, 0.2]), 3)


def test_adaptive_metropolis_structured_multichain_run():
    def model(obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None):
        del obs_times, obs_values, ctrl_times, ctrl_values
        numpyro.sample("a", dist.Normal(0.0, 1.0))
        numpyro.sample("b", dist.Normal(jnp.zeros(2), 1.0).to_event(1))

    inference = MCMCInference(
        AdaptiveMetropolisConfig(
            num_samples=4,
            num_warmup=2,
            num_chains=2,
            initial_proposal_scale=jnp.array([0.2, 0.3, 0.4]),
            init_strategy=init_to_value(values={"a": 0.0, "b": jnp.zeros(2)}),
        ),
        model,
    )
    samples = inference.run(jr.PRNGKey(2), jnp.zeros(1), jnp.zeros(1))

    assert samples["a"].shape == (2, 4)
    assert samples["b"].shape == (2, 4, 2)
    assert inference.get_diagnostics()["final_proposal_scale"].shape == (2, 3)


def test_numpyro_nuts_target_acceptance_and_diagnostics(monkeypatch):
    seen = {}
    original_nuts = mcmc_module.NUTS

    def recording_nuts(*args, **kwargs):
        seen["target_accept_prob"] = kwargs["target_accept_prob"]
        return original_nuts(*args, **kwargs)

    monkeypatch.setattr(mcmc_module, "NUTS", recording_nuts)
    inference = MCMCInference(
        NUTSConfig(
            num_samples=2,
            num_warmup=3,
            num_chains=1,
            mcmc_source="numpyro",
            target_acceptance_rate=0.91,
        ),
        _normal_model,
    )
    inference.run(jr.PRNGKey(3), jnp.zeros(1), jnp.zeros(1))

    assert seen["target_accept_prob"] == 0.91
    assert set(inference.get_diagnostics()) == {
        "mean_acceptance_rate",
        "num_divergences",
    }


def test_blackjax_nuts_target_acceptance_and_diagnostics(monkeypatch):
    seen = {}
    original_window_adaptation = blackjax_mcmc_module.blackjax.window_adaptation

    def recording_window_adaptation(*args, **kwargs):
        seen["target_acceptance_rate"] = kwargs["target_acceptance_rate"]
        return original_window_adaptation(*args, **kwargs)

    monkeypatch.setattr(
        blackjax_mcmc_module.blackjax,
        "window_adaptation",
        recording_window_adaptation,
    )
    inference = MCMCInference(
        NUTSConfig(
            num_samples=2,
            num_warmup=3,
            num_chains=1,
            mcmc_source="blackjax",
            target_acceptance_rate=0.92,
        ),
        _normal_model,
    )
    inference.run(jr.PRNGKey(4), jnp.zeros(1), jnp.zeros(1))

    assert seen["target_acceptance_rate"] == 0.92
    assert set(inference.get_diagnostics()) == {
        "mean_acceptance_rate",
        "num_divergences",
    }
