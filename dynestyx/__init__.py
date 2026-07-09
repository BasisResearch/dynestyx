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
from dynestyx.inference.state_paths.layout import (
    prepare_latent_path_layout,
)
from dynestyx.models import (
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
    ObservationModel,
    ScalarDiffusion,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.observation_missingness import prepare_missing_observation_metadata
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
    "GaussianStateEvolution",
    "Discretizer",
    "ObservationModel",
    "Filter",
    "LatentPathBuilder",
    "prepare_latent_path_layout",
    "prepare_missing_observation_metadata",
    "Smoother",
    "flatten_draws",
    "condition",
    "ConditionedResult",
    "SimulatedResult",
    "log_prob",
    "plate",
    "sample",
    "simulate",
    "DiracIdentityObservation",
    "LinearGaussianObservation",
    "LinearGaussianObservationParams",
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
