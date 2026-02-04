import jax
from typing import Union, Dict, Protocol, Optional
import numpyro.distributions as dist
import dataclasses
import warnings
import equinox as eqx

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

        if state_dim is None:
            raise ValueError(
                "state_dim is required; auto-infer is not implemented yet."
            )
        if observation_dim is None:
            raise ValueError(
                "observation_dim is required; auto-infer is not implemented yet."
            )
        if control_dim is None:
            control_dim = 0
            warnings.warn(
                "control_dim is not provided; auto-infer is not implemented yet. Setting to 0."
            )

        self.state_dim: int = state_dim
        self.observation_dim: int = observation_dim
        self.control_dim: int = control_dim


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


class ObservationModel(eqx.Module):
    """p(y_t | State_t, Control_t, t)"""

    def log_prob(self, y, x=None, u=None, t=None, *args, **kwargs):
        dist = self(x, u, t)
        return dist.log_prob(y)

    def sample(self, x, u, t, *args, **kwargs):
        dist = self(x, u, t)
        if "seed" in kwargs:  # for CD-Dynamax compatibility
            seed = kwargs.pop("seed")
            kwargs["key"] = seed
        return dist.sample(*args, **kwargs)


# Control Model
class ControlModel(eqx.Module):
    """
    u_t ~ p(u_t | State_t, t)
    Deterministic controls should use dist.Delta.
    """

    pass


class DiscreteTimeStateEvolution(StateEvolution):
    """
    x_{t+1} ~ p(x_{t+1} | State_t, Control_t, t)
    Return a NumPyro Distribution over next state.
    """

    def __call__(
        self,
        x: State,
        u: Optional[Control],
        t_now: Time,
        t_next: Time,
    ) -> dist.Distribution:
        raise NotImplementedError()
