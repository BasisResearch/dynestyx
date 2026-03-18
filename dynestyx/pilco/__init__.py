"""PILCO: Probabilistic Inference for Learning COntrol.

JAX/Equinox implementation of the PILCO algorithm (Deisenroth & Rasmussen, 2011)
integrated with the dynestyx dynamical systems framework.

Key components:

- ``MGPR``: Multi-output GP regression with analytic moment matching
- ``GPStateEvolution``: Dynestyx ``DiscreteTimeStateEvolution`` wrapping the GP
- ``MomentMatchingPropagator``: Effectful handler for ``dsx.sample`` that
  propagates Gaussian beliefs via moment matching instead of sampling
- ``PILCO``: Main algorithm orchestrating model learning, trajectory
  prediction, and policy optimization
- ``LinearController`` / ``RBFController``: Parametric policies with
  analytic action distributions under uncertain states
- ``ExponentialReward``: Saturating reward with analytic expected value

References:
    Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A Model-Based and
    Data-Efficient Approach to Policy Search. ICML.
"""

from dynestyx.pilco.controllers import LinearController, RBFController, squash_sin
from dynestyx.pilco.envs import InvertedPendulumEnv
from dynestyx.pilco.mgpr import MGPR, GPStateEvolution
from dynestyx.pilco.pilco import PILCO, MomentMatchingPropagator
from dynestyx.pilco.rewards import ExponentialReward

__all__ = [
    "MGPR",
    "GPStateEvolution",
    "PILCO",
    "MomentMatchingPropagator",
    "LinearController",
    "RBFController",
    "squash_sin",
    "ExponentialReward",
    "InvertedPendulumEnv",
]
