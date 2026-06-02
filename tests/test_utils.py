"""Utility functions for tests."""

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from numpyro.infer import Predictive

_OUTPUT_MASTER_DIR: Path | None = None


def assert_tree_all_finite(tree, *, where: str = "value") -> None:
    """Assert that every floating-point leaf in a nested value is finite."""

    def _walk(node, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                next_path = f"{path}.{key}" if path else f".{key}"
                _walk(value, next_path)
            return

        if isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                _walk(value, f"{path}[{index}]")
            return

        try:
            arr = jnp.asarray(node)
        except Exception:
            return

        if not jnp.issubdtype(arr.dtype, jnp.inexact):
            return

        finite = jnp.isfinite(arr)
        if bool(jnp.all(finite)):
            return

        bad_index = tuple(int(i) for i in np.argwhere(~np.asarray(finite))[0])
        bad_value = np.asarray(arr)[bad_index]
        location = f"{where}{path}" if path else where
        raise AssertionError(
            f"{location} contains non-finite value {bad_value} at index {bad_index}"
        )

    _walk(tree, "")


def assert_trace_sites_exist_and_field_all_finite(
    tr,
    *site_names: str,
    field: str = "value",
    where: str,
) -> None:
    """Assert that selected trace sites exist and have a finite field."""

    selected = {}
    for site_name in site_names:
        if site_name not in tr:
            raise AssertionError(f"{where}: site {site_name!r} not found in trace")

        site = tr[site_name]
        if field not in site:
            raise AssertionError(
                f"{where}: site {site_name!r} does not contain field {field!r}"
            )

        selected[site_name] = site[field]

    assert_tree_all_finite(
        selected,
        where=where,
    )


def test_assert_trace_sites_exist_and_field_all_finite_missing_site_error():
    tr = {"present": {"value": jnp.array(1.0)}}

    with pytest.raises(
        AssertionError,
        match=r"demo trace: site 'missing' not found in trace",
    ):
        assert_trace_sites_exist_and_field_all_finite(
            tr,
            "present",
            "missing",
            where="demo trace",
        )


def test_assert_trace_sites_exist_and_field_all_finite_nonfinite_error():
    tr = {"present": {"value": jnp.array([1.0, jnp.nan])}}

    with pytest.raises(
        AssertionError,
        match=r"demo trace\.present contains non-finite value nan at index \(1,\)",
    ):
        assert_trace_sites_exist_and_field_all_finite(
            tr,
            "present",
            where="demo trace",
        )


def run_profile_likelihood(
    model: Callable,
    param_name: str,
    true_val: float,
    param_min: float,
    param_max: float,
    n_grid: int = 21,
    output_dir: Path | None = None,
    output_name: str = "profile_likelihood.png",
    rng_key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute profile log-likelihood over a parameter grid using Predictive.

    Uses Predictive with params to fix the profiled parameter at each grid point,
    runs the model, and extracts f_marginal_loglik from the output.

    Args:
        model: The data-conditioned model (must use Filter).
        param_name: Name of the parameter to profile.
        true_val: True parameter value (for vertical line in plot).
        param_min: Minimum of parameter grid.
        param_max: Maximum of parameter grid.
        n_grid: Number of grid points.
        output_dir: If set, save the profile plot here.
        output_name: Filename for the saved plot.
        rng_key: Random key for Predictive.

    Returns:
        (param_grid, profile_values) as JAX arrays.

    Raises:
        ValueError: If model output does not contain f_marginal_loglik.
    """
    if rng_key is None:
        rng_key = jr.PRNGKey(0)

    param_grid = jnp.linspace(param_min, param_max, n_grid)
    keys = jr.split(rng_key, n_grid)

    def get_mll(param_val: float, key: jax.Array) -> float:
        pred = Predictive(
            model,
            params={param_name: jnp.array(param_val)},
            num_samples=1,
            exclude_deterministic=False,
        )
        out = pred(key)
        if "f_marginal_loglik" not in out:
            raise ValueError(
                "Model does not have f_marginal_loglik site. "
                "Profile likelihood requires Filter."
            )
        return float(out["f_marginal_loglik"].squeeze())

    profile = jnp.array([get_mll(float(param_grid[i]), keys[i]) for i in range(n_grid)])

    if output_dir is not None:
        import matplotlib.pyplot as plt

        profile_arr = np.array(profile)
        ymin, ymax = float(profile_arr.min()), float(profile_arr.max())
        dynamic_range = ymax - ymin
        ymean = float(np.mean(profile_arr))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(np.array(param_grid), profile_arr, label="Profile log-likelihood")
        ax.axvline(
            true_val,
            color="gray",
            linestyle="--",
            label=f"True {param_name} = {true_val}",
        )
        ax.set_xlabel(param_name)
        ax.set_ylabel(r"$\log p(y_{1:T} \mid " + param_name + ")$")

        # Choose scale from dynamic range: linear for flat profiles, symlog/log for wide range
        if dynamic_range < 10000 or (
            abs(ymean) > 1e-10 and dynamic_range / abs(ymean) < 0.1
        ):
            ax.set_yscale("linear")
        else:
            ax.set_yscale("symlog")

        ylow = ymin - 0.2 * dynamic_range  # 20% below ymin
        yhigh = ymax + 0.2 * dynamic_range  # 20% above ymax
        ax.set_ylim(ylow, yhigh)
        ax.legend()
        ax.set_title(f"Profile likelihood: {param_name}")
        plt.tight_layout()
        plt.savefig(output_dir / output_name, dpi=150, bbox_inches="tight")
        plt.close()

    return param_grid, profile


def get_output_dir(test_name: str) -> Path:
    """Create and return an output directory for a test within the master output directory.

    All calls within a test session share a common first directory (e.g. .output/20260217_102927)
    so that outputs from a single pytest run are grouped together.

    Args:
        test_name: Name of the test (e.g., "test_discreteTime_generic")

    Returns:
        Path to the output directory (e.g., ".output/20260217_102927/test_discreteTime_generic")
    """
    master_dir_str = os.environ.get("TEST_OUTPUT_MASTER_DIR", None)
    if master_dir_str is not None:
        master_dir = Path(master_dir_str)
    else:
        global _OUTPUT_MASTER_DIR
        if _OUTPUT_MASTER_DIR is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            _OUTPUT_MASTER_DIR = Path(".output") / timestamp
            _OUTPUT_MASTER_DIR.mkdir(parents=True, exist_ok=True)
        master_dir = _OUTPUT_MASTER_DIR

    output_dir = master_dir / test_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
