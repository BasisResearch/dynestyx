"""
Smoke tests for MCMC inference on discrete-time models using FilterBasedMarginalLogLikelihood.

This module tests MCMC inference pipelines for discrete-time models using
the new discrete-time filtering capabilities via bootstrap particle filters.
"""

import jax.random as jr
from numpyro.infer import MCMC, NUTS

from tests.fixtures import (
    data_conditioned_discrete_time_l63_filter,  # noqa: F401
)

NUM_SAMPLES = 10
NUM_WARMUP = 10


def test_discrete_time_l63_taylor_kf_mcmc_smoke(
    data_conditioned_discrete_time_l63_filter,  # noqa: F811
):
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
