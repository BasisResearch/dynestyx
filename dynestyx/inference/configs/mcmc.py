import dataclasses
from collections.abc import Callable
from typing import Literal

import numpy as np
from jax.typing import ArrayLike
from numpyro.infer.initialization import init_to_sample

MCMCSource = Literal["numpyro", "blackjax"]


@dataclasses.dataclass
class BaseMCMCConfig:
    """Shared configuration options inherited by all MCMC configs.

    You do not instantiate this class directly; use one of the concrete
    subclasses (`NUTSConfig`, `HMCConfig`, `AdaptiveMetropolisConfig`,
    `SGLDConfig`, `MALAConfig`, `AdjustedMCLMCDynamicConfig`).

    Attributes:
        num_samples (int): Number of post-warmup samples to return.
        num_warmup (int): Number of warmup/burn-in transitions.
        num_chains (int): Number of Markov chains to run in parallel.
        mcmc_source (MCMCSource): Backend library used for inference.
            Supported values are `"numpyro"` and `"blackjax"`.
        init_strategy (callable): NumPyro initialization strategy used when
            constructing unconstrained initial parameters.
    """

    num_samples: int
    num_warmup: int
    num_chains: int
    mcmc_source: MCMCSource
    init_strategy: Callable = init_to_sample


@dataclasses.dataclass
class HMCConfig(BaseMCMCConfig):
    """Hamiltonian Monte Carlo (HMC) configuration.

    Attributes:
        step_size (float): Integrator step size used by the leapfrog solver.
            Ignored when ``adapt=True`` (step size is tuned during warmup).
        num_steps (int): Number of leapfrog steps per HMC proposal.
        adapt (bool): Whether to tune step size and mass matrix during warmup.
            Defaults to ``True``. Set to ``False`` to use a fixed ``step_size``
            and identity mass matrix (useful when warmup is expensive or the
            step size is known in advance).
    """

    step_size: float = 1e-2
    num_steps: int = 10
    adapt: bool = True


@dataclasses.dataclass
class NUTSConfig(BaseMCMCConfig):
    """No-U-Turn Sampler (NUTS) configuration.

    Attributes:
        target_acceptance_rate (float): Target acceptance probability used
            during warmup. Must lie strictly between zero and one.
    """

    target_acceptance_rate: float = 0.8

    def __post_init__(self) -> None:
        if not 0.0 < self.target_acceptance_rate < 1.0:
            raise ValueError("target_acceptance_rate must be between 0 and 1")


@dataclasses.dataclass
class AdaptiveMetropolisConfig(BaseMCMCConfig):
    """Adaptive componentwise random-walk Metropolis configuration.

    One transition updates each flattened unconstrained coordinate in order.
    Proposal scales adapt during warmup toward the requested acceptance rate
    and remain fixed while retained samples are generated.

    This sampler is currently implemented by the BlackJAX integration only.

    Attributes:
        initial_proposal_scale (ArrayLike): Positive scalar proposal scale, or
            one positive scale per flattened unconstrained coordinate.
        target_acceptance_rate (float): Per-coordinate target acceptance rate.
        adaptation_rate (float): Exponent in the diminishing adaptation step
            ``n_iter ** -adaptation_rate``.
        max_adaptation (float): Maximum change to a log proposal scale in one
            warmup transition.
    """

    mcmc_source: MCMCSource = "blackjax"
    initial_proposal_scale: ArrayLike = 1.0
    target_acceptance_rate: float = 0.44
    adaptation_rate: float = 0.5
    max_adaptation: float = 0.01

    def __post_init__(self) -> None:
        if self.mcmc_source != "blackjax":
            raise ValueError(
                "AdaptiveMetropolisConfig only supports mcmc_source='blackjax'"
            )
        if not 0.0 < self.target_acceptance_rate < 1.0:
            raise ValueError("target_acceptance_rate must be between 0 and 1")
        if self.adaptation_rate <= 0.0:
            raise ValueError("adaptation_rate must be positive")
        if self.max_adaptation <= 0.0:
            raise ValueError("max_adaptation must be positive")

        proposal_scale = np.asarray(self.initial_proposal_scale)
        if proposal_scale.ndim > 1:
            raise ValueError("initial_proposal_scale must be a scalar or 1D array")
        if proposal_scale.size == 0 or not np.all(np.isfinite(proposal_scale)):
            raise ValueError("initial_proposal_scale must contain finite values")
        if np.any(proposal_scale <= 0.0):
            raise ValueError("initial_proposal_scale must be positive")


@dataclasses.dataclass
class SGLDConfig(BaseMCMCConfig):
    r"""Stochastic Gradient Langevin Dynamics (SGLD) configuration.

    SGLD performs first-order Langevin updates using noisy gradients and
    injected Gaussian noise. In this implementation, gradients are computed
    on the full dataset (no minibatching), so the method behaves as
    full-batch Langevin dynamics with an annealed step schedule.

    Attributes:
        step_size (float): Base learning rate used in the SGLD schedule.
            This should generally be small.
        schedule_power (float): Power in the polynomial decay schedule
            \(\epsilon_t = \text{step_size} \cdot t^{-\text{schedule_power}}\).
            Values in `(0.5, 1.0]` are common for asymptotic convergence.
    """

    step_size: float = 1e-4
    schedule_power: float = 0.55


@dataclasses.dataclass
class MALAConfig(BaseMCMCConfig):
    """Metropolis-Adjusted Langevin Algorithm (MALA) configuration.

    Attributes:
        step_size (float): Proposal step size used by `blackjax.mala`.
    """

    step_size: float = 1e-2


@dataclasses.dataclass
class AdjustedMCLMCDynamicConfig(BaseMCMCConfig):
    """Dynamic adjusted MCLMC (MHMCHMC) configuration.

    This maps to `blackjax.adjusted_mclmc_dynamic(...)` and uses its
    top-level API arguments.

    Attributes:
        step_size (float): Integrator step size.
        L_proposal_factor (float): Proposal length scaling factor.
        divergence_threshold (float): Energy-difference threshold used to flag
            divergences.
        integration_steps_min (int): Minimum random integration steps per
            proposal.
        integration_steps_max (int): Exclusive upper bound for random
            integration steps per proposal.
    """

    step_size: float = 1e-2
    L_proposal_factor: float = float("inf")
    divergence_threshold: float = 1000.0
    integration_steps_min: int = 1
    integration_steps_max: int = 10
