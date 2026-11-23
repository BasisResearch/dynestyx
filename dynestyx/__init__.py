"""Dynestyx package."""

import warnings
from importlib.metadata import version

__version__ = version("dynestyx")

# Load the model and handler primitives before configs that use shared types.
from dynestyx.handlers import condition, plate, sample

# isort: split
from dynestyx.api import log_prob, simulate
from dynestyx.discretizers import (
    Discretizer,
    discretize_dynamics,
    discretize_state_evolution,
)
from dynestyx.evaluation import Evaluation
from dynestyx.inference.filters import Filter
from dynestyx.inference.latent.builder import LatentPathBuilder
from dynestyx.inference.smoothers import Smoother
from dynestyx.models import (
    AffineDrift,
    ContinuousTimeStateEvolution,
    Covariance,
    DeterministicContinuousTimeStateEvolution,
    DeterministicObservation,
    DeterministicStateEvolution,
    DiagonalCovariance,
    DiagonalDiffusion,
    Diffusion,
    DiracIdentityObservation,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    FullCovariance,
    FullDiffusion,
    GaussianObservation,
    GaussianStateEvolution,
    ImExDrift,
    Layout,
    LayoutCollection,
    LinearGaussianObservation,
    LinearGaussianObservationParams,
    LinearGaussianParams,
    LinearGaussianStateEvolution,
    LTI_continuous,
    LTI_discrete,
    ObservationControlAlignment,
    ObservationModel,
    ScalarCovariance,
    ScalarDiffusion,
    StochasticContinuousTimeStateEvolution,
    linearize_drift,
)
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    masked_observation_log_prob,
    prepare_missing_observation_metadata,
)
from dynestyx.prepared import PreparedModel
from dynestyx.relaxations import GaussianRelaxation, relax_dynamics
from dynestyx.simulation import (
    DiscreteTimeSimulator,
    ODESimulator,
    SDESimulator,
    Simulator,
)
from dynestyx.types import (
    ConditionedResult,
    EvaluationResult,
    FilterModelInputs,
    LatentStateResult,
    Result,
    SimulatedResult,
    StructuredSimulatedResult,
)
from dynestyx.utils import flatten_draws

from . import configs

_DEPRECATED_CONFIG_EXPORTS = (
    "ObservationScoringConfig",
    "ODESimulatorConfig",
    "SDESimulatorConfig",
    "SimulatorConfig",
)


def __getattr__(name: str):
    if name in _DEPRECATED_CONFIG_EXPORTS:
        warnings.warn(
            f"dynestyx.{name} is deprecated; use dynestyx.configs.{name} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(configs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "configs",
    "Covariance",
    "ScalarCovariance",
    "DiagonalCovariance",
    "FullCovariance",
    "DeterministicStateEvolution",
    "DeterministicObservation",
    "GaussianRelaxation",
    "relax_dynamics",
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
    "Layout",
    "LayoutCollection",
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
    "PreparedModel",
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
    "FilterModelInputs",
    "LatentStateResult",
    "Result",
    "SimulatedResult",
    "StructuredSimulatedResult",
    "log_prob",
    "plate",
    "prepare_missing_observation_metadata",
    "sample",
    "simulate",
    "DiracIdentityObservation",
    "LinearGaussianObservation",
    "LinearGaussianObservationParams",
    "GaussianObservation",
    "DiscreteTimeSimulator",
    "ODESimulator",
    "SDESimulator",
    "Simulator",
    *_DEPRECATED_CONFIG_EXPORTS,
]
