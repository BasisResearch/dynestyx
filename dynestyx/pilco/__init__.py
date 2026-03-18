"""PILCO: Probabilistic Inference for Learning COntrol (Deisenroth & Rasmussen, 2011)."""

from dynestyx.pilco.controllers import LinearController, RBFController, squash_sin
from dynestyx.pilco.envs import InvertedPendulumEnv
from dynestyx.pilco.mgpr import MGPR, GPStateEvolution
from dynestyx.pilco.pilco import (
    PILCO,
    MomentMatchingPropagator,
    collect_random_rollout,
    collect_rollout,
)
from dynestyx.pilco.rewards import ExponentialReward

__all__ = [
    "MGPR",
    "GPStateEvolution",
    "PILCO",
    "MomentMatchingPropagator",
    "collect_rollout",
    "collect_random_rollout",
    "LinearController",
    "RBFController",
    "squash_sin",
    "ExponentialReward",
    "InvertedPendulumEnv",
]
