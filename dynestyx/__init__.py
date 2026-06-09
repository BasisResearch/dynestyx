"""Dynestyx package."""

from importlib.metadata import version

__version__ = version("dynestyx")

from dynestyx.discretizers import Discretizer, euler_maruyama
from dynestyx.handlers import plate, sample
from dynestyx.inference.filters import Filter
from dynestyx.inference.smoothers import Smoother
from dynestyx.models import (
    WSP,
    Box,
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
    LinearGaussianStateEvolution,
    LTI_continuous,
    LTI_discrete,
    ObservationModel,
    ScalarDiffusion,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.simulators import (
    DiscreteTimeSimulator,
    ODESimulator,
    SDESimulator,
    Simulator,
)
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
    "LinearGaussianStateEvolution",
    "GaussianStateEvolution",
    "Discretizer",
    "ObservationModel",
    "Filter",
    "Smoother",
    "flatten_draws",
    "plate",
    "sample",
    "DiracIdentityObservation",
    "LinearGaussianObservation",
    "GaussianObservation",
    "DiscreteTimeSimulator",
    "ODESimulator",
    "SDESimulator",
    "Simulator",
    "euler_maruyama",
    "WSP",
    "Box",
]
