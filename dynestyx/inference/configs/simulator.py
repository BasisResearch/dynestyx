"""Deprecated compatibility module for :mod:`dynestyx.configs.simulator`."""

import warnings

from dynestyx.configs.simulator import *  # noqa: F403

warnings.warn(
    "dynestyx.inference.configs.simulator is deprecated; use "
    "dynestyx.configs.simulator instead.",
    DeprecationWarning,
    stacklevel=2,
)
