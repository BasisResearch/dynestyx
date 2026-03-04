import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import Predictive

from dynestyx.inference.filter_configs import ContinuousTimeEnKFConfig
from dynestyx.inference.mcmc import FilterBasedMCMC
from dynestyx.inference.mcmc_configs import SGLDConfig
from dynestyx.simulators import Simulator
from tests.models import continuous_time_deterministic_l63_model


@pytest.mark.parametrize("num_samples", [120])
def test_sgmcmc_inference(num_samples):
    obs_times = jnp.arange(start=0.0, stop=2.0, step=0.01)
    true_params = {"rho": jnp.array(28.0)}
    predictive = Predictive(
        continuous_time_deterministic_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with Simulator():
        synthetic = predictive(jr.PRNGKey(0), obs_times=obs_times)
    obs_values = synthetic["observations"].squeeze(0)

    inference = FilterBasedMCMC(
        filter_config=ContinuousTimeEnKFConfig(),
        mcmc_config=SGLDConfig(
            num_samples=num_samples,
            num_warmup=num_samples,
            num_chains=1,
            mcmc_source="blackjax",
            step_size=8e-5,
            schedule_power=0.6,
        ),
        model=continuous_time_deterministic_l63_model,
    )
    posterior_samples = inference.run(jr.PRNGKey(1), obs_times, obs_values)

    posterior_rho = posterior_samples["rho"][0]
    assert posterior_rho.shape[0] == num_samples
    assert not jnp.isnan(posterior_rho).any()
    assert not jnp.isinf(posterior_rho).any()
    assert jnp.abs(posterior_rho.mean() - true_params["rho"]) < 10.0
