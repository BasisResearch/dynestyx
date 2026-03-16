"""Science tests for LTI_discrete with SGMCMC."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import Predictive

from dynestyx.inference.filters import Filter
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import SGLDConfig
from dynestyx.simulators import DiscreteTimeSimulator
from tests.fixtures import _squeeze_sim_dims
from tests.models import discrete_time_lti_simplified_model


@pytest.mark.parametrize("num_samples", [160])
def test_sgmcmc_inference(num_samples):
    predict_times = jnp.arange(start=0.0, stop=60.0, step=1.0)
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
    obs_values = _squeeze_sim_dims(synthetic["f_observations"])

    with Filter():
        inference = MCMCInference(
            mcmc_config=SGLDConfig(
                num_samples=num_samples,
                num_warmup=num_samples,
                num_chains=1,
                mcmc_source="blackjax",
                step_size=5e-5,
                schedule_power=0.6,
            ),
            model=discrete_time_lti_simplified_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(1), obs_times, obs_values)

    posterior_alpha = posterior_samples["alpha"][0]
    assert posterior_alpha.shape[0] == num_samples
    assert not jnp.isnan(posterior_alpha).any()
    assert not jnp.isinf(posterior_alpha).any()
    assert jnp.abs(posterior_alpha.mean() - true_params["alpha"]) < 0.5
