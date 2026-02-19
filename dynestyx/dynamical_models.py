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


class AffineDrift(eqx.Module):
    """
    Affine drift: f(x, u, t) = A @ x + B @ u + b.
    """

    A: jax.Array
    B: jax.Array | None = None
    b: jax.Array | None = None

    def __call__(
        self,
        x: State,
        u: Control | None,
        t: Time,
    ) -> dState:
        out = jnp.dot(self.A, x)
        if self.B is not None:
            u_vec = u if u is not None else jnp.zeros(self.B.shape[1])
            out = out + jnp.dot(self.B, u_vec)
        if self.b is not None:
            out = out + self.b
        return out


@dataclasses.dataclass
class ContinuousTimeStateEvolution:
    """
    SDE: dx = f(State_t, t) dt + L(State_t, t) dW
    """

    drift: Drift | None = None
    diffusion_coefficient: Drift | None = None
    diffusion_covariance: Drift | None = None

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


def LTI_discrete(
    A: jax.Array,
    Q: jax.Array,
    H: jax.Array,
    R: jax.Array,
    B: jax.Array | None = None,
    b: jax.Array | None = None,
    D: jax.Array | None = None,
    d: jax.Array | None = None,
    initial_mean: jax.Array | None = None,
    initial_cov: jax.Array | None = None,
) -> DynamicalModel:
    """
    Build a discrete-time LTI DynamicalModel from core generating parameters.

    State:  x_{t+1} ~ N(A x_t + B u_t + b, Q)
    Obs:    y_t ~ N(H x_t + D u_t + d, R)

    Compatible with KF, EKF, UKF via LinearGaussianStateEvolution and
    LinearGaussianObservation; also compatible with PF.

    Args:
        A: State transition matrix, shape (state_dim, state_dim).
        Q: Process noise covariance, shape (state_dim, state_dim).
        H: Observation matrix, shape (observation_dim, state_dim).
        R: Observation noise covariance, shape (observation_dim, observation_dim).
        B: Control input matrix, shape (state_dim, control_dim). If None, control_dim=0.
        b: State evolution bias, shape (state_dim,). If None, zero bias.
        D: Observation control matrix, shape (observation_dim, control_dim). If None, no control in obs.
        d: Observation bias, shape (observation_dim,). If None, zero bias.
        initial_mean: Initial state mean, shape (state_dim,). If None, zeros.
        initial_cov: Initial state covariance, shape (state_dim, state_dim). If None, identity.
    """
    state_dim = A.shape[0]
    observation_dim = H.shape[0]
    control_dim = B.shape[1] if B is not None else 0

    if initial_mean is None:
        initial_mean = jnp.zeros(state_dim)
    if initial_cov is None:
        initial_cov = jnp.eye(state_dim)

    initial_condition = dist.MultivariateNormal(
        loc=initial_mean, covariance_matrix=initial_cov
    )
    state_evolution = LinearGaussianStateEvolution(A=A, cov=Q, B=B, bias=b)
    from dynestyx.observations import LinearGaussianObservation

    observation_model = LinearGaussianObservation(H=H, R=R, D=D, bias=d)

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_model=None,
        state_dim=state_dim,
        observation_dim=observation_dim,
        control_dim=control_dim,
        continuous_time=False,
    )


def LTI_continuous(
    A: jax.Array,
    L: jax.Array,
    H: jax.Array,
    R: jax.Array,
    bm_dim: int,
    B: jax.Array | None = None,
    b: jax.Array | None = None,
    D: jax.Array | None = None,
    d: jax.Array | None = None,
    initial_mean: jax.Array | None = None,
    initial_cov: jax.Array | None = None,
) -> DynamicalModel:
    """
    Build a continuous-time LTI DynamicalModel from core generating parameters.

    SDE:  dx = (A x + B u + b) dt + L dW_t,   dW_t ~ N(0, I dt)
    Obs:  y_t ~ N(H x_t + D u_t + d, R)

    L is the diffusion coefficient (not a covariance). It maps the Brownian
    increment dW_t into the state space: L has shape (state_dim, bm_dim),
    where bm_dim is the dimension of the driving Brownian motion dW_t.
    With standard Brownian (dW_t ~ N(0, I dt)), the state noise covariance
    over dt is L @ L.T * dt. Common choices: L square (state_dim, state_dim)
    for independent noise per state; or L column vector for scalar noise.

    Args:
        A: Drift matrix, shape (state_dim, state_dim).
        L: Diffusion coefficient, shape (state_dim, bm_dim). Maps
            bm_dim-dimensional dW_t into state_dim-dimensional noise.
        H: Observation matrix, shape (observation_dim, state_dim).
        R: Observation noise covariance, shape (observation_dim, observation_dim).
        bm_dim: Dimension of the driving Brownian motion dW_t. Must equal L.shape[1].
        B: Control input matrix, shape (state_dim, control_dim). If None, control_dim=0.
        b: Drift bias, shape (state_dim,). If None, zero bias.
        D: Observation control matrix, shape (observation_dim, control_dim). If None, no control in obs.
        d: Observation bias, shape (observation_dim,). If None, zero bias.
        initial_mean: Initial state mean, shape (state_dim,). If None, zeros.
        initial_cov: Initial state covariance, shape (state_dim, state_dim). If None, identity.
    """
    state_dim = A.shape[0]
    observation_dim = H.shape[0]
    control_dim = B.shape[1] if B is not None else 0
    if L.shape[1] != bm_dim:
        raise ValueError(
            f"L.shape[1]={L.shape[1]} does not match bm_dim={bm_dim}. "
            "L must have shape (state_dim, bm_dim)."
        )

    if initial_mean is None:
        initial_mean = jnp.zeros(state_dim)
    if initial_cov is None:
        initial_cov = jnp.eye(state_dim)

    initial_condition = dist.MultivariateNormal(
        loc=initial_mean, covariance_matrix=initial_cov
    )

    drift = AffineDrift(A=A, B=B, b=b)
    L_const = L
    cov_I = jnp.eye(bm_dim)

    state_evolution = ContinuousTimeStateEvolution(
        drift=drift,
        diffusion_coefficient=lambda x, u, t: L_const,
        diffusion_covariance=lambda x, u, t: cov_I,
    )

    from dynestyx.observations import LinearGaussianObservation

    observation_model = LinearGaussianObservation(H=H, R=R, D=D, bias=d)

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_model=None,
        state_dim=state_dim,
        observation_dim=observation_dim,
        control_dim=control_dim,
        continuous_time=True,
    )


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
