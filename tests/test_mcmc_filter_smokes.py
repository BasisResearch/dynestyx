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
from tests.fixtures import _squeeze_sim_dims
from tests.models import (
    continuous_time_stochastic_l63_model,
    discrete_time_lti_simplified_model,
)

SMOKE_NUM_SAMPLES = 1
SMOKE_NUM_WARMUP = 1
SMOKE_HMC_NUM_STEPS = 4


def _make_data_continuous():
    predict_times = jnp.arange(start=0.0, stop=1.0, step=0.1)
    obs_times = predict_times
    true_params = {"rho": jnp.array(28.0)}
    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with Simulator():
        synthetic = predictive(jr.PRNGKey(0), predict_times=predict_times)
    return obs_times, _squeeze_sim_dims(synthetic["f_observations"])


def _make_data_discrete():
    obs_times = jnp.arange(start=0.0, stop=8.0, step=1.0)
    true_params = {"alpha": jnp.array(0.35)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with Simulator():
        synthetic = predictive(jr.PRNGKey(0), predict_times=obs_times)
    return obs_times, _squeeze_sim_dims(synthetic["f_observations"])


def test_filter_based_mcmc_nuts_smoke():
    obs_times, obs_values = _make_data_continuous()
    with Filter():
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=SMOKE_NUM_SAMPLES,
                num_warmup=SMOKE_NUM_WARMUP,
                num_chains=1,
                mcmc_source="numpyro",
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(0), obs_times, obs_values)
    assert "rho" in posterior_samples


def test_filter_based_mcmc_hmc_smoke():
    obs_times, obs_values = _make_data_continuous()
    with Filter():
        inference = MCMCInference(
            mcmc_config=HMCConfig(
                num_samples=SMOKE_NUM_SAMPLES,
                num_warmup=SMOKE_NUM_WARMUP,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=5e-3,
                num_steps=SMOKE_HMC_NUM_STEPS,
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(1), obs_times, obs_values)
    assert "rho" in posterior_samples


def test_discrete_filter_based_mcmc_hmc_smoke():
    obs_times, obs_values = _make_data_discrete()
    with Filter():
        inference = MCMCInference(
            mcmc_config=HMCConfig(
                num_samples=SMOKE_NUM_SAMPLES,
                num_warmup=SMOKE_NUM_WARMUP,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=5e-3,
                num_steps=SMOKE_HMC_NUM_STEPS,
            ),
            model=discrete_time_lti_simplified_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(1), obs_times, obs_values)
    assert "alpha" in posterior_samples


def test_filter_based_sgmcmc_smoke():
    obs_times, obs_values = _make_data_continuous()
    with Filter():
        inference = MCMCInference(
            mcmc_config=SGLDConfig(
                num_samples=SMOKE_NUM_SAMPLES,
                num_warmup=SMOKE_NUM_WARMUP,
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
    obs_times, obs_values = _make_data_continuous()
    with Filter():
        inference = MCMCInference(
            mcmc_config=MALAConfig(
                num_samples=SMOKE_NUM_SAMPLES,
                num_warmup=SMOKE_NUM_WARMUP,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=1e-3,
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(3), obs_times, obs_values)
    assert "rho" in posterior_samples


if __name__ == "__main__":
    test_discrete_filter_based_mcmc_hmc_smoke()
