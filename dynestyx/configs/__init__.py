"""Public configuration options for Dynestyx handlers and samplers."""

from typing import TYPE_CHECKING

from dynestyx.configs.discretizer import (
    DiffraxSampleConfig,
    DiscretizerConfig,
    EulerMaruyamaConfig,
    ExactAffineConfig,
    LocalLinearizationConfig,
    MeanTrajectoryLinearizationConfig,
    ODEFlowConfig,
)
from dynestyx.configs.filters import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
    CustomFilterConfig,
    EKFConfig,
    EnKFConfig,
    EnKFLocalizationConfig,
    EnKFLocalizationFunctions,
    HMMConfig,
    KFConfig,
    PFConfig,
    PFResamplingConfig,
    UKFConfig,
)
from dynestyx.configs.mcmc import (
    AdaptiveMetropolisConfig,
    AdaptiveMWGConfig,
    HMCConfig,
    MALAConfig,
    NUTSConfig,
    SGLDConfig,
)
from dynestyx.configs.simulator import (
    ODESimulatorConfig,
    SDESimulatorConfig,
    SimulatorConfig,
)
from dynestyx.configs.smoother import (
    ContinuousTimeEKFSmootherConfig,
    ContinuousTimeKFSmootherConfig,
    EKFSmootherConfig,
    EnRTSSmootherConfig,
    KFSmootherConfig,
    PFSmootherConfig,
    UKFSmootherConfig,
)

if TYPE_CHECKING:
    from dynestyx.configs.scoring import ObservationScoringConfig


def __getattr__(name: str):
    if name == "ObservationScoringConfig":
        # The evaluation package imports the scoring config for its handler.
        from dynestyx.configs.scoring import ObservationScoringConfig

        return ObservationScoringConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "AdaptiveMetropolisConfig",
    "AdaptiveMWGConfig",
    "ContinuousTimeDPFConfig",
    "ContinuousTimeEKFConfig",
    "ContinuousTimeEKFSmootherConfig",
    "ContinuousTimeEnKFConfig",
    "ContinuousTimeKFConfig",
    "ContinuousTimeKFSmootherConfig",
    "ContinuousTimeUKFConfig",
    "CustomFilterConfig",
    "DiffraxSampleConfig",
    "DiscretizerConfig",
    "EKFConfig",
    "EKFSmootherConfig",
    "EnKFConfig",
    "EnKFLocalizationConfig",
    "EnKFLocalizationFunctions",
    "EnRTSSmootherConfig",
    "EulerMaruyamaConfig",
    "ExactAffineConfig",
    "HMCConfig",
    "HMMConfig",
    "KFConfig",
    "KFSmootherConfig",
    "LocalLinearizationConfig",
    "MALAConfig",
    "MeanTrajectoryLinearizationConfig",
    "NUTSConfig",
    "ObservationScoringConfig",
    "ODEFlowConfig",
    "ODESimulatorConfig",
    "PFConfig",
    "PFResamplingConfig",
    "PFSmootherConfig",
    "SDESimulatorConfig",
    "SGLDConfig",
    "SimulatorConfig",
    "UKFConfig",
    "UKFSmootherConfig",
]
