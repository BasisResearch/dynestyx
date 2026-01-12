"""
Consolidated smoke tests for MCMC inference.

This module imports and runs smoke tests from the test_science modules,
ensuring that all MCMC inference pipelines can run with minimal parameters.
"""

from tests.test_science.test_discreteTime_generic import (
    test_mcmc_smoke as _test_discreteTime_generic_smoke,
)
from tests.test_science.test_hmm import test_mcmc_smoke as _test_hmm_smoke
from tests.test_science.test_l63_mcmc import test_mcmc_smoke as _test_l63_mcmc_smoke


def test_discreteTime_generic_mcmc_smoke():
    """Smoke test for discrete time generic MCMC inference."""
    _test_discreteTime_generic_smoke()


def test_hmm_mcmc_smoke():
    """Smoke test for HMM MCMC inference."""
    _test_hmm_smoke()


def test_l63_mcmc_smoke():
    """Smoke test for Lorenz 63 MCMC inference."""
    _test_l63_mcmc_smoke()
