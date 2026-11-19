"""Deprecated compatibility module for :mod:`dynestyx.configs.mcmc`."""

import warnings

from dynestyx.configs.mcmc import *  # noqa: F403

warnings.warn(
    "dynestyx.inference.configs.mcmc is deprecated; use dynestyx.configs.mcmc instead.",
    DeprecationWarning,
    stacklevel=2,
)
