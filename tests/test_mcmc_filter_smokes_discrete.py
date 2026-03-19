"""Smoke tests for MCMCInference on discrete-time models."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import Predictive

from dynestyx.inference.filters import Filter
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import NUTSConfig, SGLDConfig
from dynestyx.simulators import DiscreteTimeSimulator
from tests.fixtures import _squeeze_sim_dims
from tests.models import discrete_time_lti_simplified_model


def _make_data():
    predict_times = jnp.arange(start=0.0, stop=30.0, step=1.0)
    obs_times = predict_times
    true_params = {"alpha": jnp.array(0.35)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(jr.PRNGKey(0), predict_times=predict_times)
    return obs_times, _squeeze_sim_dims(synthetic["f_observations"])


def test_filter_based_discrete_nuts_smoke():
    obs_times, obs_values = _make_data()
    with Filter():
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=10, num_warmup=10, num_chains=1, mcmc_source="numpyro"
            ),
            model=discrete_time_lti_simplified_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(0), obs_times, obs_values)
    assert "alpha" in posterior_samples


@pytest.mark.parametrize("num_chains", [1])
def test_filter_based_discrete_sgmcmc_smoke(num_chains):
    obs_times, obs_values = _make_data()
    with Filter():
        inference = MCMCInference(
            mcmc_config=SGLDConfig(
                num_samples=10,
                num_warmup=10,
                num_chains=num_chains,
                mcmc_source="blackjax",
                step_size=5e-5,
                schedule_power=0.6,
            ),
            model=discrete_time_lti_simplified_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(1), obs_times, obs_values)
    assert "alpha" in posterior_samples
