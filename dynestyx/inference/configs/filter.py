"""Deprecated compatibility module for :mod:`dynestyx.configs.filter`."""

import warnings

from dynestyx.configs.filter import *  # noqa: F403

warnings.warn(
    "dynestyx.inference.configs.filter is deprecated; use "
    "dynestyx.configs.filter instead.",
    DeprecationWarning,
    stacklevel=2,
)
