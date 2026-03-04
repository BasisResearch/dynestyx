"""Smoke tests for FilterBasedMCMC on discrete-time models."""

import jax.numpy as jnp
import jax.random as jr
from numpyro.infer import Predictive

from dynestyx.inference.filter_configs import KFConfig
from dynestyx.inference.mcmc import FilterBasedMCMC
from dynestyx.inference.mcmc_configs import NUTSConfig, SGLDConfig
from dynestyx.simulators import DiscreteTimeSimulator
from tests.models import discrete_time_lti_simplified_model


def _make_data():
    obs_times = jnp.arange(start=0.0, stop=30.0, step=1.0)
    true_params = {"alpha": jnp.array(0.35)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(jr.PRNGKey(0), obs_times=obs_times)
    return obs_times, synthetic["observations"].squeeze(0)


def test_filter_based_discrete_nuts_smoke():
    obs_times, obs_values = _make_data()
    inference = FilterBasedMCMC(
        filter_config=KFConfig(filter_source="cd_dynamax"),
        mcmc_config=NUTSConfig(
            num_samples=10, num_warmup=10, num_chains=1, mcmc_source="numpyro"
        ),
        model=discrete_time_lti_simplified_model,
    )
    posterior_samples = inference.run(jr.PRNGKey(0), obs_times, obs_values)
    assert "alpha" in posterior_samples


def test_filter_based_discrete_sgmcmc_smoke():
    obs_times, obs_values = _make_data()
    inference = FilterBasedMCMC(
        filter_config=KFConfig(filter_source="cd_dynamax"),
        mcmc_config=SGLDConfig(
            num_samples=10,
            num_warmup=10,
            num_chains=1,
            mcmc_source="blackjax",
            step_size=5e-5,
            schedule_power=0.6,
        ),
        model=discrete_time_lti_simplified_model,
    )
    posterior_samples = inference.run(jr.PRNGKey(1), obs_times, obs_values)
    assert "alpha" in posterior_samples
