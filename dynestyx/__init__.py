"""Dynestyx package."""

from importlib.metadata import version

__version__ = version("dynestyx")

from dynestyx.api import log_prob, simulate
from dynestyx.discretizers import Discretizer, euler_maruyama
from dynestyx.handlers import condition, plate, sample
from dynestyx.inference.configs.simulator import (
    ODESimulatorConfig,
    SDESimulatorConfig,
    SimulatorConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.inference.latent.builder import LatentPathBuilder
from dynestyx.inference.smoothers import Smoother
from dynestyx.models import (
    AffineDrift,
    ContinuousTimeStateEvolution,
    DeterministicContinuousTimeStateEvolution,
    DiagonalDiffusion,
    Diffusion,
    DiracIdentityObservation,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    GaussianObservation,
    GaussianStateEvolution,
    LinearGaussianObservation,
    LinearGaussianObservationParams,
    LinearGaussianParams,
    LinearGaussianStateEvolution,
    LTI_continuous,
    LTI_discrete,
    MixedStateDistribution,
    ObservationModel,
    ScalarDiffusion,
    StochasticContinuousTimeStateEvolution,
    SwitchingLinearGaussianObservation,
    SwitchingLinearGaussianStateEvolution,
)
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    prepare_missing_observation_metadata,
)
from dynestyx.simulation import (
    DiscreteTimeSimulator,
    ODESimulator,
    SDESimulator,
    Simulator,
)
from dynestyx.types import ConditionedResult, SimulatedResult
from dynestyx.utils import flatten_draws

__all__ = [
    "__version__",
    "ContinuousTimeStateEvolution",
    "DeterministicContinuousTimeStateEvolution",
    "Diffusion",
    "FullDiffusion",
    "DiagonalDiffusion",
    "ScalarDiffusion",
    "StochasticContinuousTimeStateEvolution",
    "DiscreteTimeStateEvolution",
    "DynamicalModel",
    "AffineDrift",
    "LTI_continuous",
    "LTI_discrete",
    "LinearGaussianParams",
    "LinearGaussianStateEvolution",
    "MixedStateDistribution",
    "GaussianStateEvolution",
    "Discretizer",
    "ObservationModel",
    "Filter",
    "LatentPathBuilder",
    "MissingObservationMetadata",
    "Smoother",
    "flatten_draws",
    "condition",
    "ConditionedResult",
    "SimulatedResult",
    "log_prob",
    "plate",
    "prepare_missing_observation_metadata",
    "sample",
    "simulate",
    "DiracIdentityObservation",
    "LinearGaussianObservation",
    "LinearGaussianObservationParams",
    "SwitchingLinearGaussianObservation",
    "SwitchingLinearGaussianStateEvolution",
    "GaussianObservation",
    "ODESimulatorConfig",
    "SDESimulatorConfig",
    "SimulatorConfig",
    "DiscreteTimeSimulator",
    "ODESimulator",
    "SDESimulator",
    "Simulator",
    "euler_maruyama",
]
