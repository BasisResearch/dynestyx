"""
Smoke tests for MCMC inference on discrete-time models using Filter.

This module tests MCMC inference pipelines for discrete-time models using
the new discrete-time filtering capabilities via bootstrap particle filters.
"""

import jax
import jax.random as jr
import pytest
from numpyro.infer import MCMC, NUTS, BarkerMH

from tests.fixtures import (
    data_conditioned_discrete_time_l63_filter,  # noqa: F401
    data_conditioned_discrete_time_l63_filter_pf,  # noqa: F401
    data_conditioned_discrete_time_lti_kf,  # noqa: F401
    data_conditioned_discrete_time_lti_simplified,  # noqa: F401
)

NUM_SAMPLES = 1
NUM_WARMUP = 1


@pytest.fixture(scope="module", autouse=True)
def config():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def test_discrete_time_l63_taylor_kf_mcmc_smoke(
    data_conditioned_discrete_time_l63_filter,  # noqa: F811
) -> None:
    """Test MCMC inference on discrete-time L63 model using Taylor-linearized Kalman filter."""
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _ = (
        data_conditioned_discrete_time_l63_filter
    )
    mcmc = MCMC(
        NUTS(data_conditioned_model), num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples


def test_discrete_time_l63_pf_mcmc_smoke(
    data_conditioned_discrete_time_l63_filter_pf,  # noqa: F811
) -> None:
    """Test MCMC inference on discrete-time L63 model using a particle filter."""
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _ = (
        data_conditioned_discrete_time_l63_filter_pf
    )
    mcmc = MCMC(
        BarkerMH(
            data_conditioned_model,
            adapt_step_size=False,
            adapt_mass_matrix=False,
        ),
        num_samples=NUM_SAMPLES,
        num_warmup=NUM_WARMUP,
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "rho" in posterior_samples


def test_discrete_time_lti_kf_mcmc_smoke(
    data_conditioned_discrete_time_lti_kf,  # noqa: F811
) -> None:
    """Test MCMC inference on discrete-time LTI model using Kalman filter (filter_type='kf')."""
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _, _ = (
        data_conditioned_discrete_time_lti_kf
    )
    mcmc = MCMC(
        NUTS(data_conditioned_model), num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "alpha" in posterior_samples


def test_discrete_time_lti_simplified_kf_mcmc_smoke(
    data_conditioned_discrete_time_lti_simplified,  # noqa: F811
) -> None:
    """Test MCMC inference on discrete-time LTI model using LTI_discrete factory + KF."""
    mcmc_key = jr.PRNGKey(0)
    data_conditioned_model, true_params, synthetic, _ = (
        data_conditioned_discrete_time_lti_simplified
    )
    mcmc = MCMC(
        NUTS(data_conditioned_model), num_samples=NUM_SAMPLES, num_warmup=NUM_WARMUP
    )
    mcmc.run(mcmc_key)
    posterior_samples = mcmc.get_samples()
    assert "alpha" in posterior_samples
