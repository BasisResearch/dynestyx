"""Core interfaces and base classes for dynamical models."""

import dataclasses
import warnings
from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
from numpyro._typing import DistributionT

from dynestyx.types import Control, State, Time, dState


class DynamicalModel(eqx.Module):
    """
    Unified interface:
        - initial_condition: DistributionT
        - state_evolution: Callable[[State, Control, Time], State] | Callable[[State, Control, Time, Time], State]
        - observation_model: Callable[[State, Control, Time], DistributionT]
        - control_model: Any
    """

    state_dim: int
    observation_dim: int
    control_dim: int
    initial_condition: DistributionT
    state_evolution: (
        Callable[[State, Control, Time], State]
        | Callable[[State, Control, Time, Time], State]
    )
    observation_model: Callable[[State, Control, Time], DistributionT]
    control_model: Any
    continuous_time: bool

    def __init__(
        self,
        initial_condition,
        state_evolution,
        observation_model,
        control_model=None,
        state_dim: int | None = None,
        observation_dim: int | None = None,
        control_dim: int | None = None,
        continuous_time: bool = False,
    ):
        if isinstance(state_evolution, ContinuousTimeStateEvolution):
            self.continuous_time = True
        else:
            self.continuous_time = False

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

        if isinstance(state_evolution, ContinuousTimeStateEvolution):
            if state_evolution.diffusion_coefficient is not None:
                if state_evolution.bm_dim is None:
                    self.state_evolution.bm_dim = state_dim  # type: ignore[union-attr]


class Drift(Protocol):
    """
    A callable mapping:
        (state, control, time) -> dState
    """

    def __call__(
        self,
        x: State,
        u: Control | None,
        t: Time,
    ) -> dState:
        raise NotImplementedError()


@dataclasses.dataclass
class ContinuousTimeStateEvolution:
    """
    SDE: dx = f(State_t, t) dt + L(State_t, t) dW
    """

    drift: Drift | None = None
    diffusion_coefficient: Drift | None = None
    bm_dim: int | None = None

    ...


class DiscreteTimeStateEvolution:
    """
    x_{t+1} ~ p(x_{t+1} | State_t, Control_t, t)
    Return a NumPyro Distribution over next state.
    """

    def __call__(
        self,
        x: State,
        u: Control | None,
        t_now: Time,
        t_next: Time,
    ) -> DistributionT:
        raise NotImplementedError()


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
