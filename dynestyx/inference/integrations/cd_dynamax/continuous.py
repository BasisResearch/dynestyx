"""Compatibility wrapper for cd-dynamax continuous-time filtering/smoothing."""

from dynestyx.inference.integrations.cd_dynamax.continuous_filter import (
    ContinuousTimeFilterConfig,
    compute_continuous_filter,
    run_continuous_filter,
)
from dynestyx.inference.integrations.cd_dynamax.continuous_smoother import (
    compute_continuous_smoother,
    run_continuous_smoother,
)

__all__ = [
    "ContinuousTimeFilterConfig",
    "compute_continuous_filter",
    "run_continuous_filter",
    "compute_continuous_smoother",
    "run_continuous_smoother",
]
