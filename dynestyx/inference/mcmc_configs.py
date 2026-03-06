import dataclasses
from collections.abc import Callable
from typing import Literal

from numpyro.infer.initialization import init_to_sample

MCMCSource = Literal["numpyro", "blackjax"]


@dataclasses.dataclass
class BaseMCMCConfig:
    """Shared configuration options inherited by all MCMC configs.

    You do not instantiate this class directly; use one of the concrete
    subclasses (`NUTSConfig`, `HMCConfig`, `SGLDConfig`).

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
        num_steps (int): Number of leapfrog steps per HMC proposal.
    """

    step_size: float = 1e-2
    num_steps: int = 10


@dataclasses.dataclass
class NUTSConfig(BaseMCMCConfig):
    """No-U-Turn Sampler (NUTS) configuration."""


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
