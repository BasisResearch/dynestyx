"""Deprecated compatibility module for :mod:`dynestyx.configs.smoother`."""

import warnings

from dynestyx.configs.smoother import *  # noqa: F403
from dynestyx.configs.smoother import SmootherConfig as SmootherConfig
from dynestyx.configs.smoother import __all__ as __all__

warnings.warn(
    "dynestyx.inference.configs.smoother is deprecated; use "
    "dynestyx.configs.smoother instead.",
    DeprecationWarning,
    stacklevel=2,
)
