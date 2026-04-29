"""Dynestyx package."""

from importlib.metadata import version

__version__ = version("dynestyx")

from dynestyx.discretizers import Discretizer, euler_maruyama
from dynestyx.handlers import plate, sample
from dynestyx.inference.filters import Filter
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DiscreteTimeStateEvolution,
    DynamicalModel,
    GaussianObservation,
    GaussianStateEvolution,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
    LTI_continuous,
    LTI_discrete,
    ObservationModel,
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
]
