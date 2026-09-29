"""ArviZ >= 1.0 helpers shared by the science tests.

All knowledge of the ArviZ return schemas is concentrated here so that a
future ArviZ release change only requires updating this module (guarded at
runtime by ``tests/test_arviz_utils.py``, which runs in the default CI job).
"""

from pathlib import Path

import numpy as np


def _as_draws_1d(draws) -> np.ndarray:
    """Validate and convert a sample of posterior draws to a 1-D numpy array."""
    array = np.asarray(draws)
    if array.ndim != 1:
        raise ValueError(
            f"Expected a 1-D array of posterior draws, got shape {array.shape}"
        )
    return array


def hdi_bounds(draws, prob: float = 0.95) -> tuple[float, float]:
    """Highest-density interval ``(lower, upper)`` for a 1-D array of draws.

    Wraps ``arviz_stats.hdi``, which returns a plain ``[lower, upper]`` array
    for raw 1-D input.
    """
    from arviz_stats import hdi

    lower, upper = np.asarray(hdi(_as_draws_1d(draws), prob=prob))
    return float(lower), float(upper)


def save_posterior_plot(
    draws,
    *,
    name: str,
    output_path: Path,
    prob: float = 0.95,
    ref_val: float | None = None,
    dpi: int = 150,
) -> None:
    """Plot a 1-D posterior sample with its HDI and save it to ``output_path``.

    Replacement for the ArviZ 0.x ``az.plot_posterior`` + ``plt.savefig``
    idiom: the draws are wrapped into a single-chain ``DataTree`` and rendered
    with ``az.plot_dist``, optionally marking a reference value.
    """
    import arviz as az
    import matplotlib.pyplot as plt

    dt = az.from_dict({"posterior": {name: _as_draws_1d(draws)[None, :]}})
    pc = az.plot_dist(
        dt,
        kind="kde",
        ci_kind="hdi",
        ci_prob=prob,
        point_estimate="mean",
    )
    if ref_val is not None:
        ax = pc.viz["plot"][name].item()
        ax.axvline(float(ref_val), color="C1", ls="--", lw=1.5, label="true value")
        ax.legend()
    fig = pc.viz["figure"].item()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
