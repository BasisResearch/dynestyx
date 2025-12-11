import jax
from typing import Union, Dict, Protocol, Optional
import numpyro.distributions as dist
import dataclasses

# ----------------------------------------------------------------------
# TYPE ALIASES
# ----------------------------------------------------------------------
State = Union[jax.Array, Dict[str, jax.Array]]
dState = State
Observation = jax.Array
Control = Optional[State]
Time = float
Params = Dict[str, Union[float, jax.Array]]
Key = jax.Array

class DynamicalModel:
    """
    Unified interface:
        - initial_condition: InitialCondition
        - state_evolution: StateEvolution (CT/DT/SDE/ODE)
        - observation_model: ObservationModel
        - control_model: optional
    """
    def __init__(
        self,
        initial_condition,
        state_evolution,
        observation_model,
        control_model=None,
        state_dim: Optional[int] = None,
        observation_dim: Optional[int] = None,
        control_dim: Optional[int] = None,
    ):
        self.initial_condition = initial_condition
        self.state_evolution = state_evolution
        self.observation_model = observation_model
        self.control_model = control_model
        
        self.state_dim = state_dim
        self.observation_dim = observation_dim
        self.control_dim = control_dim
        # TODO: auto-infer dims from models.

class InitialCondition(dist.Distribution):
    """
    The initial-condition is a distribution over State.
    """
    pass

class StateEvolution:
    """
    Base class: DT or CT or SDE.
    Contains only:
        - drift / transition functions
        - diffusion for SDE
    No stepping. No sampling.
    """
    pass
    ...

class Drift(Protocol):
    """
    A callable mapping:
        (state, control, time) -> dState
    """

    def __call__(
        self,
        x: State,
        u: Optional[Control],
        t: Time,
    ) -> dState:
        raise NotImplementedError()

@dataclasses.dataclass
class ContinuousTimeStateEvolution(StateEvolution):
    """
    SDE: dx = f(State_t, t) dt + L(State_t, t) dW
    """

    drift: Optional[Drift] = None
    diffusion_coefficient: Optional[Drift] = None
    diffusion_covariance: Optional[Drift] = None
        
    ...

class DistributionFromStateTimeParams(Protocol):
    """
    A callable mapping:
        (state, time, params) -> numpyro.distributions.Distribution

    Used for:
        - ObservationModel
        - DiscreteTimeStateEvolution
        - ControlModel

    This is a structural type: anything with this __call__ signature is valid.
    """
    def __call__(
        self,
        x: State,
        u: Optional[Control],
        t: Time,
    ) -> dist.Distribution:
        ...

class ObservationModel(DistributionFromStateTimeParams):
    """p(y_t | State_t, Control_t, t)"""
    pass


# Control Model
class ControlModel(DistributionFromStateTimeParams):
    """
    u_t ~ p(u_t | State_t, t)
    Deterministic controls should use dist.Delta.
    """
    pass

class DiscreteTimeStateEvolution(StateEvolution, DistributionFromStateTimeParams):
    """
    x_{t+1} ~ p(x_{t+1} | State_t, Control_t, t)
    Return a NumPyro Distribution over next state.
    """

    def __call__(
        self,
        x: State,
        u: Optional[Control],
        t: Time,
    ) -> dist.Distribution:
        raise NotImplementedError()
    
