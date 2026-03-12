import jax.numpy as jnp
import jax.random as jr
import pytest

from dynestyx.inference.filter_configs import ContinuousTimeEnKFConfig, EKFConfig
from dynestyx.inference.filters import Filter
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import SGLDConfig
from tests.models import continuous_time_stochastic_l63_model, discrete_time_l63_model


@pytest.mark.parametrize(
    ("model", "filter_config", "target"),
    [
        (continuous_time_stochastic_l63_model, ContinuousTimeEnKFConfig(), "rho"),
        (discrete_time_l63_model, EKFConfig(filter_source="cuthbert"), "rho"),
    ],
)
def test_sgmcmc_inference_smoke(model, filter_config, target):
    """Smoke test version - minimal samples to verify code runs without errors."""
    obs_times = jnp.arange(start=0.0, stop=1.0, step=0.1)
    obs_values = jnp.zeros((obs_times.shape[0], 1))

    with Filter(filter_config):
        inference = MCMCInference(
            mcmc_config=SGLDConfig(
                num_samples=5,
                num_warmup=5,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=1e-4,
                schedule_power=0.55,
            ),
            model=model,
        )
        posterior_samples = inference.run(
            rng_key=jr.PRNGKey(0),
            obs_times=obs_times,
            obs_values=obs_values,
        )

    assert target in posterior_samples
    values = posterior_samples[target]
    assert values.shape[0] == 1
    assert values.shape[1] == 5
    assert not jnp.isnan(values).any()
    assert not jnp.isinf(values).any()


def test_sgmcmc_inference_smoke_multiple_chains():
    """Smoke test with multiple chains to verify chain axis handling."""
    obs_times = jnp.arange(start=0.0, stop=1.0, step=0.1)
    obs_values = jnp.zeros((obs_times.shape[0], 1))

    with Filter():
        inference = MCMCInference(
            mcmc_config=SGLDConfig(
                num_samples=4,
                num_warmup=4,
                num_chains=2,
                mcmc_source="blackjax",
                step_size=1e-4,
                schedule_power=0.55,
            ),
            model=continuous_time_stochastic_l63_model,
        )
        posterior_samples = inference.run(
            rng_key=jr.PRNGKey(1),
            obs_times=obs_times,
            obs_values=obs_values,
        )

    assert "rho" in posterior_samples
    values = posterior_samples["rho"]
    assert values.shape[0] == 2
    assert values.shape[1] == 4
    assert not jnp.isnan(values).any()
    assert not jnp.isinf(values).any()
