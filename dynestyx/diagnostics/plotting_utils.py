"""Deprecated compatibility path for evaluation plotting utilities.

Use :mod:`dynestyx.evaluation.plotting_utils` instead. This module will be
removed in v0.4.0.
"""

import warnings

from dynestyx.evaluation.plotting_utils import (
    plot_continuous_states_and_partial_observations,
    plot_drift_field,
    plot_hmm_states_and_observations,
)

warnings.warn(
    "`dynestyx.diagnostics.plotting_utils` is deprecated; use "
    "`dynestyx.evaluation.plotting_utils` instead. The deprecated import path "
    "will be removed in v0.5.0.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "plot_continuous_states_and_partial_observations",
    "plot_drift_field",
    "plot_hmm_states_and_observations",
]
