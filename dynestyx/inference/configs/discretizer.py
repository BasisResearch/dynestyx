"""Deprecated compatibility module for :mod:`dynestyx.configs.discretizer`."""

import warnings

from dynestyx.configs.discretizer import *  # noqa: F403

warnings.warn(
    "dynestyx.inference.configs.discretizer is deprecated; use "
    "dynestyx.configs.discretizer instead.",
    DeprecationWarning,
    stacklevel=2,
)
