"""Smoother options shared across backends.

`dynestyx` smoothers reuse existing filter config classes for algorithm selection.
This module provides lightweight smoother-specific options for backend details
that do not belong in filter configs.
"""

import dataclasses
from typing import Literal

PFBackwardSamplingMethod = Literal["tracing", "exact", "mcmc"]
CDKFSmootherType = Literal["cd_smoother_1", "cd_smoother_2"]


@dataclasses.dataclass
class SmootherOptions:
    """Backend-specific smoothing options.

    Attributes:
        pf_backward_sampling_method: Backward simulation method for cuthbert PF
            smoothing. Defaults to `"tracing"`.
        pf_mcmc_n_steps: Number of IMH steps when using
            `pf_backward_sampling_method="mcmc"`.
        pf_n_smoother_particles: Number of smoother particles for PF smoothing.
            If `None`, inherits `PFConfig.n_particles`.
        cdlgssm_smoother_type: Continuous-discrete Kalman smoother variant used
            by cd-dynamax for `ContinuousTimeKFConfig`.
    """

    pf_backward_sampling_method: PFBackwardSamplingMethod = "tracing"
    pf_mcmc_n_steps: int = 10
    pf_n_smoother_particles: int | None = None
    cdlgssm_smoother_type: CDKFSmootherType = "cd_smoother_1"
