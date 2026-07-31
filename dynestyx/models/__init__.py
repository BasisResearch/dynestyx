"""Dynamical models: core interfaces, state evolution, and observations.

Structure anticipates future extension to LTI factories, Neural SDEs, etc.
"""

from dynestyx.distributions import MixedStateDistribution
from dynestyx.models.core import (
    ContinuousTimeStateEvolution,
    DeterministicContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    Drift,
    DynamicalModel,
    ObservationModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.diffusions import (
    DiagonalDiffusion,
    Diffusion,
    FullDiffusion,
    ScalarDiffusion,
)
from dynestyx.models.lti_dynamics import LTI_continuous, LTI_discrete
from dynestyx.models.observations import (
    DiracIdentityObservation,
    GaussianObservation,
    LinearGaussianObservation,
    LinearGaussianObservationParams,
    SwitchingLinearGaussianObservation,
)
from dynestyx.models.state_evolution import (
    AffineDrift,
    GaussianStateEvolution,
    LinearGaussianParams,
    LinearGaussianStateEvolution,
    SwitchingLinearGaussianStateEvolution,
)

__all__ = [
    "ContinuousTimeStateEvolution",
    "DeterministicContinuousTimeStateEvolution",
    "AffineDrift",
    "DiracIdentityObservation",
    "Diffusion",
    "DiscreteTimeStateEvolution",
    "DiagonalDiffusion",
    "DynamicalModel",
    "Drift",
    "FullDiffusion",
    "GaussianObservation",
    "GaussianStateEvolution",
    "LinearGaussianObservation",
    "LinearGaussianObservationParams",
    "LinearGaussianParams",
    "LinearGaussianStateEvolution",
    "MixedStateDistribution",
    "ObservationModel",
    "SwitchingLinearGaussianObservation",
    "SwitchingLinearGaussianStateEvolution",
    "StochasticContinuousTimeStateEvolution",
    "LTI_continuous",
    "LTI_discrete",
    "ScalarDiffusion",
]
