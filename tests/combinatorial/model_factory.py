from typing import Callable

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx.dynamical_models import Trajectory

from tests.combinatorial.specs import DataSpec, ModelSpec


def _is_scalar(x) -> bool:
    return jnp.ndim(x) == 0


def _state_size(x) -> int:
    if _is_scalar(x):
        return 1
    return int(x.shape[0])


def _control_size(u) -> int:
    if _is_scalar(u):
        return 1
    return int(u.shape[0])


def _build_control_map_matrix(x, u):
    x_dim = _state_size(x)
    u_dim = _control_size(u)
    vals = jnp.arange(1, x_dim * u_dim + 1, dtype=jnp.float32)
    return vals.reshape((x_dim, u_dim)) / float(x_dim * u_dim)


def _project_control_to_state(x, u):
    """Project controls into state space using deterministic B(x_dim, u_dim).

    This is intentionally written in a simple "user-style" way, with minimal
    dimension handling, to surface backend shape handling gaps early.
    """
    if u is None:
        return 0 if _is_scalar(x) else jnp.zeros_like(x)

    B = _build_control_map_matrix(x, u)
    if _is_scalar(x) and _is_scalar(u):
        return B[0, 0] * u
    if _is_scalar(x):
        return (B @ u)[0]
    if _is_scalar(u):
        return B[:, 0] * u
    return B @ u


class LinearDiscreteTransition(eqx.Module):
    uses_control: bool

    def __call__(self, x, u, t_now, t_next):
        dt = t_next - t_now
        u_term = _project_control_to_state(x, u) if (self.uses_control and u is not None) else 0
        loc = x + dt * (x + u_term)
        if jnp.ndim(loc) == 0:
            return dist.Normal(loc=loc, scale=1)
        return dist.MultivariateNormal(
            loc=loc, covariance_matrix=jnp.eye(loc.shape[-1])
        )


class NonlinearDiscreteTransition(eqx.Module):
    uses_control: bool

    def __call__(self, x, u, t_now, t_next):
        dt = t_next - t_now
        nonlin = jnp.sin(x)
        u_term = _project_control_to_state(x, u) if (self.uses_control and u is not None) else 0
        loc = x + dt * (nonlin + u_term)
        if jnp.ndim(loc) == 0:
            return dist.Normal(loc=loc, scale=1)
        return dist.MultivariateNormal(
            loc=loc, covariance_matrix=jnp.eye(loc.shape[-1])
        )


class ContinuousZeroDriftTransition(eqx.Module):
    uses_control: bool

    def __call__(self, x, u, t):
        # Keep continuous-time drift fixed at zero for these combinatorial tests.
        drift = jnp.zeros_like(x)
        u_term = _project_control_to_state(x, u) if (self.uses_control and u is not None) else 0
        return drift + u_term


def time_grid(timesteps: int):
    if timesteps == 1:
        return jnp.array([0.0])
    return jnp.linspace(0.0, 1.0, timesteps)


def make_data(data: DataSpec):
    times = time_grid(data.timesteps)
    obs_dim = 1 if data.obs_rank == 1 else 2
    if data.obs_rank == 1:
        obs_values = jnp.linspace(0.0, 1.0, data.timesteps)
    else:
        obs_values = jnp.stack(
            [jnp.linspace(0.0, 1.0, data.timesteps), jnp.linspace(1.0, 2.0, data.timesteps)],
            axis=-1,
        )

    if data.ctrl_rank == 0:
        controls = Trajectory()
    elif data.ctrl_rank == 1:
        controls = Trajectory(times=times, values=jnp.ones((data.timesteps,)))
    else:
        controls = Trajectory(times=times, values=jnp.ones((data.timesteps, 2)))

    return times, obs_values, controls, obs_dim


def _make_initial_condition(kind: str, init_rank: int):
    state_dim = 1 if init_rank == 1 else 2
    if kind == "mvn":
        loc = jnp.zeros((state_dim,))
        cov = jnp.eye(state_dim)
        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov), state_dim
    if kind == "uniform":
        low = -jnp.ones((state_dim,))
        high = jnp.ones((state_dim,))
        return dist.Uniform(low, high).to_event(1), state_dim

    if init_rank == 1:
        probs = jnp.array([0.4, 0.6])
    else:
        probs = jnp.array([[0.3, 0.7], [0.8, 0.2]])
    return dist.Categorical(probs=probs), state_dim


def _make_observation_model(kind: str, obs_rank: int, state_dim: int) -> Callable:
    obs_dim = 1 if obs_rank == 1 else 2
    if kind == "perfect":
        return dsx.DiracIdentityObservation(), obs_dim
    if kind == "linear_gaussian":
        H = jnp.eye(obs_dim, state_dim)
        R = jnp.eye(obs_dim)
        return dsx.LinearGaussianObservation(H=H, R=R), obs_dim

    def poisson_obs(x, u, t):
        if obs_dim == 1:
            rate = jnp.exp(jnp.sum(x))
        else:
            if x.shape[0] >= 2:
                rate = jnp.exp(x[:2])
            else:
                rate = jnp.exp(jnp.repeat(x, 2))
        return dist.Poisson(rate)

    return poisson_obs, obs_dim


def _make_discrete_transition(transition_kind: str, uses_control: bool):
    if transition_kind == "categorical":

        def fn(x, u, t_now, t_next):
            probs = jnp.array([[0.85, 0.15], [0.2, 0.8]])
            return dist.Categorical(probs=probs[x])

        return fn

    if transition_kind == "linear_mvn":
        return LinearDiscreteTransition(uses_control=uses_control)
    return NonlinearDiscreteTransition(uses_control=uses_control)


def _make_continuous_transition(
    uses_control: bool,
    transition_kind: str,
    diffusion_coeff: str,
    diffusion_cov: str,
):
    _ = transition_kind
    drift = ContinuousZeroDriftTransition(uses_control=uses_control)

    dcoeff = None
    if diffusion_coeff == "eye":
        dcoeff = lambda x, u, t: jnp.eye(_state_size(x))
    dcov = None
    if diffusion_cov == "eye":
        dcov = lambda x, u, t: jnp.eye(_state_size(x))
    return dsx.ContinuousTimeStateEvolution(
        drift=drift,
        diffusion_coefficient=dcoeff,
        diffusion_covariance=dcov,
    )


def build_model(spec: ModelSpec):
    initial_condition, state_dim = _make_initial_condition(spec.initial_kind, spec.init_rank)
    obs_model, obs_dim = _make_observation_model(
        spec.observation_kind, spec.observation_rank, state_dim
    )

    if spec.family == "continuous":
        state_evolution = _make_continuous_transition(
            uses_control=spec.uses_control,
            transition_kind=spec.transition_kind,
            diffusion_coeff=spec.diffusion_coeff,
            diffusion_cov=spec.diffusion_cov,
        )
    else:
        state_evolution = _make_discrete_transition(
            transition_kind=spec.transition_kind,
            uses_control=spec.uses_control,
        )

    control_dim = 0 if not spec.uses_control else 2
    dynamics = dsx.DynamicalModel(
        state_dim=state_dim,
        observation_dim=obs_dim,
        control_dim=control_dim,
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=obs_model,
    )

    def model():
        dsx.sample("f", dynamics)

    return model, dynamics

