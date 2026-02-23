import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx.models.core import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
)
from dynestyx.models.observations import LinearGaussianObservation
from dynestyx.models.state_evolution import AffineDrift, LinearGaussianStateEvolution


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
    control_dim = B.shape[1] if B is not None else 0

    if initial_mean is None:
        initial_mean = jnp.zeros(state_dim)
    elif initial_mean.shape != (state_dim,):
        raise ValueError(
            f"initial_mean must have shape ({state_dim},), got {initial_mean.shape}"
        )
    if initial_cov is None:
        initial_cov = jnp.eye(state_dim)
    elif initial_cov.shape != (state_dim, state_dim):
        raise ValueError(
            f"initial_cov must have shape ({state_dim}, {state_dim}), got {initial_cov.shape}"
        )

    initial_condition = dist.MultivariateNormal(
        loc=initial_mean, covariance_matrix=initial_cov
    )
    state_evolution = LinearGaussianStateEvolution(A=A, cov=Q, B=B, bias=b)

    observation_model = LinearGaussianObservation(H=H, R=R, D=D, bias=d)

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_model=None,
        control_dim=control_dim,
    )


def LTI_continuous(
    A: jax.Array,
    L: jax.Array,
    H: jax.Array,
    R: jax.Array,
    B: jax.Array | None = None,
    b: jax.Array | None = None,
    D: jax.Array | None = None,
    d: jax.Array | None = None,
    initial_mean: jax.Array | None = None,
    initial_cov: jax.Array | None = None,
) -> DynamicalModel:
    f"""
    Build a continuous-time LTI DynamicalModel from core generating parameters.

    SDE:  dx = (A x(t) + B u(t) + b) dt + L dW_t,   dW_t ~ N(0, I_{L.shape[1]} dt)
    Obs:  y_t ~ N(H x_t + D u_t + d, R)

    L is the diffusion coefficient (not a covariance). It maps the Brownian
    increment dW_t into the state space: L has shape (state_dim, L.shape[1]),
    where L.shape[1] is the dimension of the driving Brownian motion dW_t.
    With standard Brownian (dW_t ~ N(0, I_{L.shape[1]} dt)), the state noise covariance
    over dt is L @ L.T * dt.

    Args:
        A: Drift matrix, shape (state_dim, state_dim).
        L: Diffusion coefficient, shape (state_dim, bm_dim). Maps
            bm_dim-dimensional dW_t into state_dim-dimensional noise.
        H: Observation matrix, shape (observation_dim, state_dim).
        R: Observation noise covariance, shape (observation_dim, observation_dim).
        B: Control input matrix, shape (state_dim, control_dim). If None, control_dim=0.
        b: Drift bias, shape (state_dim,). If None, zero bias.
        D: Observation control matrix, shape (observation_dim, control_dim). If None, no control in obs.
        d: Observation bias, shape (observation_dim,). If None, zero bias.
        initial_mean: Initial state mean, shape (state_dim,). If None, zeros.
        initial_cov: Initial state covariance, shape (state_dim, state_dim). If None, identity.
    """
    state_dim = A.shape[0]
    control_dim = B.shape[1] if B is not None else 0

    if initial_mean is None:
        initial_mean = jnp.zeros(state_dim)
    elif initial_mean.shape != (state_dim,):
        raise ValueError(
            f"initial_mean must have shape ({state_dim},), got {initial_mean.shape}"
        )
    if initial_cov is None:
        initial_cov = jnp.eye(state_dim)
    elif initial_cov.shape != (state_dim, state_dim):
        raise ValueError(
            f"initial_cov must have shape ({state_dim}, {state_dim}), got {initial_cov.shape}"
        )

    initial_condition = dist.MultivariateNormal(
        loc=initial_mean, covariance_matrix=initial_cov
    )

    drift = AffineDrift(A=A, B=B, b=b)

    state_evolution = ContinuousTimeStateEvolution(
        drift=drift,
        diffusion_coefficient=lambda x, u, t: L,
    )

    observation_model = LinearGaussianObservation(H=H, R=R, D=D, bias=d)

    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=observation_model,
        control_model=None,
        control_dim=control_dim,
    )
