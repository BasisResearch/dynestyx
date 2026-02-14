import dataclasses
import warnings
from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro._typing import DistributionT

# ----------------------------------------------------------------------
# TYPE ALIASES
# ----------------------------------------------------------------------
State = jax.Array
dState = State
Observation = jax.Array
Control = State | None
Time = jax.Array
Params = dict[str, float | jax.Array]
Key = jax.Array
FunctionOfTime = Callable[[Time], State]


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


class LinearGaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    x_t_next | x_t_now, u_t_now, t_now, t_next ~ Normal( A x_t_now + B u_t_now + bias, cov )

    where A is the observation matrix, B is the control matrix, bias is the bias, and cov is the state noise covariance.
    """

    A: jax.Array
    B: jax.Array | None = None
    bias: jax.Array | None = None
    cov: jax.Array

    def __init__(
        self,
        A: jax.Array,
        cov: jax.Array,
        B: jax.Array | None = None,
        bias: jax.Array | None = None,
    ):
        self.A = A
        self.B = B
        self.bias = bias
        self.cov = cov

    def __call__(self, x, u, t_now, t_next):
        loc = jnp.dot(self.A, x)
        if self.bias is not None:
            loc += self.bias
        if self.B is not None and u is not None:
            loc += jnp.dot(self.B, u)

        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov)


class GaussianStateEvolution(DiscreteTimeStateEvolution):
    """
    x_t_next | x_t_now, u_t_now, t_now, t_next ~ Normal( F(x_t_now, u_t_now, t_now, t_next), cov )

    where F is a callable mapping (State, Control, Time) -> State
    and cov is the state noise covariance.
    """

    F: Callable[[State, Control, Time, Time], State]
    cov: jax.Array

    def __init__(
        self,
        F: Callable[[State, Control, Time, Time], State],
        cov: jax.Array,
    ):
        self.F = F
        self.cov = cov

    def __call__(self, x, u, t_now, t_next):
        loc = self.F(x, u, t_now, t_next)

        return dist.MultivariateNormal(loc=loc, covariance_matrix=self.cov)


@dataclasses.dataclass
class Trajectory:
    """
    A 1D time axis and values living on that axis.

    Semantics:
      - times is None  -> times are implicit / inferred / shared with some other grid
      - values is None -> "no values here" (e.g. just a solve grid)
    """

    times: Time | None = None
    values: State | None = None


@dataclasses.dataclass
class Context:
    """
    All time-indexed info for a single sample site.
    """

    # Where to solve the dynamics
    solve: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Observations
    observations: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Controls u(t), if any
    controls: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Extensible: extra time-indexed series or metadata
    extras: dict[str, Trajectory] = dataclasses.field(default_factory=dict)
