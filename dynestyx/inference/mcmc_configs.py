import dataclasses
from typing import Literal

from numpyro.infer.initialization import init_to_sample

MCMCSource = Literal["numpyro", "blackjax"]


@dataclasses.dataclass
class BaseMCMCConfig:
    num_samples: int
    num_warmup: int
    num_chains: int
    mcmc_source: MCMCSource
    init_strategy: callable = init_to_sample


@dataclasses.dataclass
class HMCConfig(BaseMCMCConfig):
    step_size: float = 1e-2
    num_steps: int = 10


@dataclasses.dataclass
class NUTSConfig(BaseMCMCConfig):
    pass
