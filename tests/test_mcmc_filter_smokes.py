"""Smoke tests for MCMCInference on continuous-time models."""

import jax.numpy as jnp
import jax.random as jr
from numpyro.infer import Predictive

from dynestyx.inference.filters import Filter
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import (
    HMCConfig,
    MALAConfig,
    NUTSConfig,
    SGLDConfig,
)
from dynestyx.simulators import Simulator
from tests.models import continuous_time_stochastic_l63_model


def _make_data():
    obs_times = jnp.arange(start=0.0, stop=2.0, step=0.05)
    true_params = {"rho": jnp.array(28.0)}
    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with Simulator():
        synthetic = predictive(jr.PRNGKey(0), obs_times=obs_times)
    return obs_times, synthetic["observations"].squeeze(0)


def test_filter_based_mcmc_nuts_smoke():
    obs_times, obs_values = _make_data()
    with Filter():
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=8, num_warmup=8, num_chains=1, mcmc_source="numpyro"
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(0), obs_times, obs_values)
    assert "rho" in posterior_samples


def test_filter_based_mcmc_hmc_smoke():
    obs_times, obs_values = _make_data()
    with Filter():
        inference = MCMCInference(
            mcmc_config=HMCConfig(
                num_samples=8,
                num_warmup=8,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=5e-3,
                num_steps=8,
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(1), obs_times, obs_values)
    assert "rho" in posterior_samples


def test_filter_based_sgmcmc_smoke():
    obs_times, obs_values = _make_data()
    with Filter():
        inference = MCMCInference(
            mcmc_config=SGLDConfig(
                num_samples=8,
                num_warmup=8,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=1e-4,
                schedule_power=0.55,
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(2), obs_times, obs_values)
    assert "rho" in posterior_samples


def test_filter_based_mala_smoke():
    obs_times, obs_values = _make_data()
    with Filter():
        inference = MCMCInference(
            mcmc_config=MALAConfig(
                num_samples=8,
                num_warmup=8,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=1e-3,
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(3), obs_times, obs_values)
    assert "rho" in posterior_samples
