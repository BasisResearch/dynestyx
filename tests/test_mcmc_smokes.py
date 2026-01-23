"""
Consolidated smoke tests for MCMC inference.

This module imports and runs smoke tests from the test_science modules,
ensuring that all MCMC inference pipelines can run with minimal parameters.
"""

import jax.random as jr
from numpyro.infer import MCMC, NUTS, BarkerMH

from tests.fixtures import (
    data_conditioned_hmm,  # noqa: F401
    data_conditioned_discrete_time_l63,  # noqa: F401
    data_conditioned_continuous_time_l63_dpf,  # noqa: F401
    data_conditioned_continuous_time_lingam,  # noqa: F401
    data_conditioned_continuous_time_lingam_dpf,  # noqa: F401
    data_conditioned_continuous_time_stochastic_l63,  # noqa: F401
    data_conditioned_continuous_time_deterministic_l63,  # noqa: F401
)

NUM_SAMPLES = 10
NUM_WARMUP = 10


def test_hmm_mcmc_smoke(data_conditioned_hmm):  # noqa: F811
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _ = data_conditioned_hmm
    mcmc = MCMC(
        NUTS(data_conditioned_model), num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "A" in posterior_samples
    assert "mu" in posterior_samples
    assert "sigma" in posterior_samples


def test_discrete_time_l63_mcmc_smoke(data_conditioned_discrete_time_l63):  # noqa: F811
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _ = (
        data_conditioned_discrete_time_l63
    )
    mcmc = MCMC(
        NUTS(data_conditioned_model), num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples


def test_continuous_time_stochastic_l63_mcmc_smoke(
    data_conditioned_continuous_time_stochastic_l63,  # noqa: F811
):
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _ = (
        data_conditioned_continuous_time_stochastic_l63
    )
    mcmc = MCMC(
        NUTS(data_conditioned_model), num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples


def test_continuous_time_deterministic_l63_mcmc_smoke(
    data_conditioned_continuous_time_deterministic_l63,  # noqa: F811
):
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _ = (
        data_conditioned_continuous_time_deterministic_l63
    )
    mcmc = MCMC(
        NUTS(data_conditioned_model), num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples


def test_continuous_time_stochastic_l63_dpf_mcmc_smoke(
    data_conditioned_continuous_time_l63_dpf,  # noqa: F811
):
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic = (
        data_conditioned_continuous_time_l63_dpf
    )
    mcmc = MCMC(BarkerMH(data_conditioned_model), num_samples=10, num_warmup=10)
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples


def test_continuous_time_lingam_mcmc_smoke(
    data_conditioned_continuous_time_lingam,  # noqa: F811
):
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, use_controls = (
        data_conditioned_continuous_time_lingam
    )
    mcmc = MCMC(NUTS(data_conditioned_model), num_samples=10, num_warmup=10)
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples


def test_continuous_time_lingam_dpf_mcmc_smoke(
    data_conditioned_continuous_time_lingam_dpf,  # noqa: F811
):
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, use_controls = (
        data_conditioned_continuous_time_lingam_dpf
    )
    mcmc = MCMC(BarkerMH(data_conditioned_model), num_samples=10, num_warmup=10)
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples
