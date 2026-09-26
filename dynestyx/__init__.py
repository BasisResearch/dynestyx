"""Dynestyx package."""

from importlib.metadata import version

__version__ = version("dynestyx")

from dynestyx.api import log_prob, simulate
from dynestyx.discretizers import (
    Discretizer,
    discretize_dynamics,
    discretize_state_evolution,
)
from dynestyx.evaluation import Evaluation, ObservationScoringConfig
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
    ImExDrift,
    LinearGaussianObservation,
    LinearGaussianObservationParams,
    LinearGaussianParams,
    LinearGaussianStateEvolution,
    LTI_continuous,
    LTI_discrete,
    ObservationControlAlignment,
    ObservationModel,
    ScalarDiffusion,
    StochasticContinuousTimeStateEvolution,
    linearize_drift,
)
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    masked_observation_log_prob,
    prepare_missing_observation_metadata,
)
from dynestyx.simulation import (
    DiscreteTimeSimulator,
    ODESimulator,
    SDESimulator,
    Simulator,
)
from dynestyx.types import ConditionedResult, EvaluationResult, SimulatedResult
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
    "ImExDrift",
    "LTI_continuous",
    "LTI_discrete",
    "LinearGaussianParams",
    "LinearGaussianStateEvolution",
    "GaussianStateEvolution",
    "Discretizer",
    "ObservationControlAlignment",
    "ObservationModel",
    "Filter",
    "Evaluation",
    "LatentPathBuilder",
    "MissingObservationMetadata",
    "Smoother",
    "flatten_draws",
    "condition",
    "masked_observation_log_prob",
    "discretize_dynamics",
    "discretize_state_evolution",
    "linearize_drift",
    "ConditionedResult",
    "EvaluationResult",
    "ObservationScoringConfig",
    "SimulatedResult",
    "log_prob",
    "plate",
    "prepare_missing_observation_metadata",
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
]
