"""Smoother configuration dataclasses.

Concrete smoother configs inherit from corresponding filter configs and expose
backend-specific smoothing options directly on the config class.
"""

import abc
import dataclasses
from typing import Literal

from dynestyx.inference.filter_configs import (
    ContinuousTimeEKFConfig,
    ContinuousTimeKFConfig,
    EKFConfig,
    KFConfig,
    PFConfig,
    UKFConfig,
)

PFBackwardSamplingMethod = Literal["tracing", "exact", "mcmc"]
CDKFSmootherType = Literal["cd_smoother_1", "cd_smoother_2"]


@dataclasses.dataclass
class SmootherConfig(abc.ABC):
    """Abstract base class for all smoother configs."""

    def __post_init__(self):
        if type(self) is SmootherConfig:
            raise TypeError("SmootherConfig is abstract and cannot be instantiated.")


@dataclasses.dataclass
class KFSmootherConfig(KFConfig, SmootherConfig):
    """Discrete-time Kalman smoother config."""


@dataclasses.dataclass
class EKFSmootherConfig(EKFConfig, SmootherConfig):
    """Discrete-time extended Kalman smoother config."""


@dataclasses.dataclass
class UKFSmootherConfig(UKFConfig, SmootherConfig):
    """Discrete-time unscented Kalman smoother config."""


@dataclasses.dataclass
class PFSmootherConfig(PFConfig, SmootherConfig):
    """Discrete-time particle smoother config."""

    pf_backward_sampling_method: PFBackwardSamplingMethod = "tracing"
    pf_mcmc_n_steps: int = 10
    pf_n_smoother_particles: int | None = None


@dataclasses.dataclass
class ContinuousTimeKFSmootherConfig(ContinuousTimeKFConfig, SmootherConfig):
    """Continuous-time Kalman smoother config."""

    cdlgssm_smoother_type: CDKFSmootherType = "cd_smoother_1"


@dataclasses.dataclass
class ContinuousTimeEKFSmootherConfig(ContinuousTimeEKFConfig, SmootherConfig):
    """Continuous-time extended Kalman smoother config."""


DiscreteTimeSmootherConfigs: tuple[type, ...] = (
    KFSmootherConfig,
    EKFSmootherConfig,
    UKFSmootherConfig,
    PFSmootherConfig,
)

ContinuousTimeSmootherConfigs: tuple[type, ...] = (
    ContinuousTimeKFSmootherConfig,
    ContinuousTimeEKFSmootherConfig,
)

__all__ = [
    "CDKFSmootherType",
    "ContinuousTimeEKFSmootherConfig",
    "ContinuousTimeKFSmootherConfig",
    "ContinuousTimeSmootherConfigs",
    "DiscreteTimeSmootherConfigs",
    "EKFSmootherConfig",
    "KFSmootherConfig",
    "PFBackwardSamplingMethod",
    "PFSmootherConfig",
    "SmootherConfig",
    "UKFSmootherConfig",
]
