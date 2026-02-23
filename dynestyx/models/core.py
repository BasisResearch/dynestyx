"""Core interfaces and base classes for dynamical models."""

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from numpyro._typing import DistributionT

from dynestyx.models.checkers import (
    _infer_vector_dim_from_distribution,
    _make_probe_state,
    _validate_state_evolution_output_shape,
)
from dynestyx.types import Control, State, Time, dState


class DynamicalModel(eqx.Module):
    """
    Unified interface:
        - initial_condition: DistributionT
        - state_evolution: Callable[[State, Control, Time], State] | Callable[[State, Control, Time, Time], State]
        - observation_model: Callable[[State, Control, Time], DistributionT]
        - control_model: Any
    """

    initial_condition: DistributionT
    state_evolution: (
        Callable[[State, Control, Time], State]
        | Callable[[State, Control, Time, Time], State]
    )
    observation_model: Callable[[State, Control, Time], DistributionT]
    control_dim: int
    control_model: Any
    state_dim: int
    observation_dim: int
    continuous_time: bool

    def __init__(
        self,
        initial_condition,
        state_evolution,
        observation_model,
        control_dim: int | None = None,
        control_model=None,
        **_internal_fields,
    ):
        # dataclasses.replace/effect handlers may pass stored fields back into
        # __init__; accept and ignore them because these are always inferred.
        _internal_fields.pop("state_dim", None)
        _internal_fields.pop("observation_dim", None)
        _internal_fields.pop("continuous_time", None)
        if _internal_fields:
            unknown = ", ".join(sorted(_internal_fields.keys()))
            raise TypeError(f"Unexpected constructor arguments: {unknown}")

        self.continuous_time = isinstance(state_evolution, ContinuousTimeStateEvolution)
        self.initial_condition = initial_condition
        self.state_evolution = state_evolution
        self.observation_model = observation_model
        self.control_model = control_model

        state_dim = _infer_vector_dim_from_distribution(
            initial_condition, "initial_condition"
        )
        if control_dim is None:
            control_dim = 0

        x0 = _make_probe_state(initial_condition=initial_condition, state_dim=state_dim)
        u0 = None if control_dim == 0 else jnp.zeros((control_dim,))
        t0 = jnp.array(0.0)

        _validate_state_evolution_output_shape(
            state_evolution=state_evolution,
            state_dim=state_dim,
            x0=x0,
            u0=u0,
            t0=t0,
            continuous_time=self.continuous_time,
        )

        obs_dist = observation_model(x0, u0, t0)
        observation_dim = _infer_vector_dim_from_distribution(
            obs_dist, "observation_model(x, u, t)"
        )

        self.state_dim = int(state_dim)
        self.observation_dim = int(observation_dim)
        self.control_dim = int(control_dim)


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


class Potential(Protocol):
    """
    A scalar potential energy callable mapping:
        (state, control, time) -> scalar potential
    """

    def __call__(
        self,
        x: State,
        u: Control | None,
        t: Time,
    ) -> jax.Array:
        raise NotImplementedError()


@dataclasses.dataclass
class ContinuousTimeStateEvolution:
    """
    SDE: dx = [drift(x, u, t) + s * grad(potential)(x, u, t)] dt + L(x, u, t) dW

    where s = -1 when `use_negative_gradient` is True, else s = +1.
    """

    drift: Drift | None = None
    potential: Potential | None = None
    use_negative_gradient: bool = False
    diffusion_coefficient: Drift | None = None
    bm_dim: int | None = dataclasses.field(default=None, repr=False)

    def total_drift(self, x: State, u: Control | None, t: Time) -> dState:
        base = self.drift(x, u, t) if self.drift is not None else None

        potential = self.potential
        if potential is None:
            if base is None:
                raise ValueError(
                    "ContinuousTimeStateEvolution requires drift or potential to be defined."
                )
            return base

        grad_potential = jax.grad(lambda z: potential(z, u, t))(x)
        sign = -1.0 if self.use_negative_gradient else 1.0
        grad_term = sign * grad_potential
        if base is None:
            return grad_term
        return base + grad_term


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
