"""Deprecated compatibility module for :mod:`dynestyx.configs.filters`."""

import warnings

from dynestyx.configs.filters import *  # noqa: F403

warnings.warn(
    "dynestyx.inference.configs.filter is deprecated; use "
    "dynestyx.configs.filters instead.",
    DeprecationWarning,
    stacklevel=2,
)
