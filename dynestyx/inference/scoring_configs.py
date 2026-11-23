"""Deprecated compatibility module for :mod:`dynestyx.configs.scoring`."""

import warnings

from dynestyx.configs.scoring import ObservationScoringConfig

__all__ = ["ObservationScoringConfig"]

warnings.warn(
    "dynestyx.inference.scoring_configs is deprecated; use "
    "dynestyx.configs.scoring instead.",
    DeprecationWarning,
    stacklevel=2,
)
