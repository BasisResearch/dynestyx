"""Tests for the ArviZ >= 1.0 helpers in ``tests/arviz_utils.py``.

These run in the default CI job and act as a runtime guard on the ArviZ
return schemas the science tests rely on.
"""

import jax.random as jr
import numpy as np
import pytest

from tests.arviz_utils import hdi_bounds, save_posterior_plot


def test_hdi_bounds_orders_and_covers():
    draws = np.random.default_rng(0).normal(size=5_000)

    lower, upper = hdi_bounds(draws, prob=0.95)
    assert lower < 0.0 < upper
    assert lower == pytest.approx(-1.96, abs=0.25)
    assert upper == pytest.approx(1.96, abs=0.25)

    lower_50, upper_50 = hdi_bounds(draws, prob=0.5)
    assert lower < lower_50 < upper_50 < upper


def test_hdi_bounds_accepts_jax_array():
    draws = jr.normal(jr.PRNGKey(0), (4_000,))

    lower, upper = hdi_bounds(draws)
    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower < upper


def test_hdi_bounds_rejects_non_1d_input():
    with pytest.raises(ValueError, match="1-D"):
        hdi_bounds(np.zeros((2, 100)))


@pytest.mark.parametrize("ref_val", [None, 0.5])
def test_save_posterior_plot_writes_file(tmp_path, ref_val):
    draws = np.random.default_rng(1).normal(loc=0.5, size=1_000)
    output_path = tmp_path / "posterior_rho.png"

    save_posterior_plot(draws, name="rho", output_path=output_path, ref_val=ref_val)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
