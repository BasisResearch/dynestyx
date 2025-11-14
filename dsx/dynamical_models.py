import jax
import jax.numpy as jnp
from typing import Union, Dict, Protocol, Optional, Callable, Any
import numpyro.distributions as dist

# ----------------------------------------------------------------------
# TYPE ALIASES
# ----------------------------------------------------------------------
State = Union[jax.Array, Dict[str, jax.Array]]
dState = State
Observation = jax.Array
Control = Optional[jax.Array]
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
    ):
        self.initial_condition = initial_condition
        self.state_evolution = state_evolution
        self.observation_model = observation_model
        self.control_model = control_model


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

class ContinuousTimeStateEvolution(StateEvolution):
    """
    ODE: dx/dt = f(State_t, t; params)
    """

    def drift(
        self,
        state: State,
        t: Time,
        params: Params
    ) -> dState:
        '''Example implementation:
            x = state['x']
            u = state['u']        
            gamma = params['gamma']
            dxdt = gamma * x + u
            return dict(x=dxdt)
        '''
        raise NotImplementedError()

class StochasticContinuousTimeStateEvolution(ContinuousTimeStateEvolution):
    """
    SDE: dx = f(State_t, t) dt + L(State_t, t) dW
    """

    def diffusion_coefficient(
        self,
        state: State,
        t: Time,
        params: Params
    ) -> jax.Array:
        """
        Example implementation:
            dim_brownian = self.diffusion_covariance(state, t, params)['x'].shape[-1]
            dim_x = state['x'].shape[-1]
            L = jnp.eye((dim_x, dim_brownian)) * params['diffusion_scale']
            return dict(x=L)
        """

        raise NotImplementedError()

    def diffusion_covariance(
        self,
        state: State,
        t: Time,
        params: Params
    ) -> jax.Array:
        """
        Example implementation:
            dim_x = state['x'].shape[-1]
            Q = jnp.eye(dim_x)
            return dict(x=Q)
            # typically is Identity
        """
        raise NotImplementedError()
        
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
        state: State,
        t: Time,
        params: Params
    ) -> dist.Distribution:
        ...

class ObservationModel(DistributionFromStateTimeParams):
    """p(y_t | State_t, t, params)"""
    pass


# Control Model
class ControlModel(DistributionFromStateTimeParams):
    """
    u_t ~ p(u_t | State_t, t, params)
    Deterministic controls should use dist.Delta.
    """
    pass

class DiscreteTimeStateEvolution(StateEvolution, DistributionFromStateTimeParams):
    """
    x_{t+1} ~ p(x_{t+1} | State_t, t, params)
    Return a NumPyro Distribution over next state.
    """

    def __call__(
        self,
        state: State,
        t: Time,
        params: Params
    ) -> dist.Distribution:
        raise NotImplementedError()
    
