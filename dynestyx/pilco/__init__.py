"""PILCO: Probabilistic Inference for Learning COntrol.

JAX/Equinox implementation of the PILCO algorithm (Deisenroth & Rasmussen, 2011)
integrated with the dynestyx dynamical systems framework.

References:
    Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A Model-Based and
    Data-Efficient Approach to Policy Search. ICML.
"""

from dynestyx.pilco.controllers import LinearController, RBFController, squash_sin
from dynestyx.pilco.envs import InvertedPendulumEnv
from dynestyx.pilco.mgpr import MGPR
from dynestyx.pilco.pilco import PILCO
from dynestyx.pilco.rewards import ExponentialReward

__all__ = [
    "MGPR",
    "PILCO",
    "LinearController",
    "RBFController",
    "squash_sin",
    "ExponentialReward",
    "InvertedPendulumEnv",
]
