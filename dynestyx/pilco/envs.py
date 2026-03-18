"""JAX environments for PILCO, producing dynestyx ``DynamicalModel`` instances."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array

from dynestyx.models.core import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
)
from dynestyx.models.observations import DiracIdentityObservation, GaussianObservation
from dynestyx.models.state_evolution import GaussianStateEvolution
from dynestyx.types import Control, State, Time


class InvertedPendulumEnv(eqx.Module):
    """
    Inverted pendulum (swing-up) environment.

    State: $[\\theta, \\dot\\theta]$, action: torque.
    Produces dynestyx ``DynamicalModel`` via ``to_continuous_dynamical_model()``
    or ``to_discrete_dynamical_model()`` for use with effectful handlers.
    """

    mass: float = 1.0
    length: float = 1.0
    g: float = 9.81
    b: float = 0.0
    dt: float = 0.05
    max_torque: float = 2.0

    def step(self, state: Array, action: Array) -> Array:
        theta, dtheta = state[0], state[1]
        u = jnp.clip(action.squeeze(), -self.max_torque, self.max_torque)
        ddtheta = (
            u - self.b * dtheta + self.mass * self.g * self.length * jnp.sin(theta)
        ) / (self.mass * self.length**2)
        new_dtheta = dtheta + ddtheta * self.dt
        new_theta = theta + new_dtheta * self.dt
        return jnp.array([new_theta, new_dtheta])

    def rollout(
        self, policy_fn, x0: Array, T: int, key: Array | None = None
    ) -> tuple[Array, Array, Array]:
        def scan_fn(state, _):
            action = policy_fn(state)
            next_state = self.step(state, action)
            return next_state, (state, action, next_state)

        _, (states, actions, next_states) = jax.lax.scan(scan_fn, x0, None, length=T)
        all_states = jnp.concatenate([states, next_states[-1:]], axis=0)
        return all_states, actions, next_states

    def random_rollout(
        self, x0: Array, T: int, key: Array
    ) -> tuple[Array, Array, Array]:
        keys = jax.random.split(key, T)

        def scan_fn(state, k):
            action = jax.random.uniform(
                k, (1,), minval=-self.max_torque, maxval=self.max_torque
            )
            next_state = self.step(state, action)
            return next_state, (state, action, next_state)

        _, (states, actions, next_states) = jax.lax.scan(scan_fn, x0, keys)
        all_states = jnp.concatenate([states, next_states[-1:]], axis=0)
        return all_states, actions, next_states

    @property
    def state_dim(self) -> int:
        return 2

    def _make_initial_condition(self, x0: Array | None = None):
        d = self.state_dim
        if x0 is not None:
            return dist.MultivariateNormal(loc=x0, covariance_matrix=0.01 * jnp.eye(d))
        return dist.MultivariateNormal(
            loc=jnp.zeros(d), covariance_matrix=0.1 * jnp.eye(d)
        )

    def to_continuous_dynamical_model(
        self, x0: Array | None = None, obs_noise: float = 0.01
    ) -> DynamicalModel:
        """``ContinuousTimeStateEvolution`` for ``ODESimulator``/``SDESimulator``."""
        mass, length, g_val, b_val = self.mass, self.length, self.g, self.b

        def drift(x: State, u: Control, t: Time) -> State:
            theta, dtheta = x[0], x[1]
            u_val = jnp.zeros(()) if u is None else u.squeeze()
            ddtheta = (
                u_val - b_val * dtheta + mass * g_val * length * jnp.sin(theta)
            ) / (mass * length**2)
            return jnp.array([dtheta, ddtheta])

        return DynamicalModel(
            initial_condition=self._make_initial_condition(x0),
            state_evolution=ContinuousTimeStateEvolution(drift=drift),
            observation_model=GaussianObservation(
                h=lambda x, u, t: x,
                R=obs_noise**2 * jnp.eye(self.state_dim),
            ),
            control_dim=1,
        )

    def to_discrete_dynamical_model(
        self, x0: Array | None = None, process_noise: float = 0.001
    ) -> DynamicalModel:
        """``GaussianStateEvolution`` for ``DiscreteTimeSimulator``/``Filter``.

        Uses dynestyx's ``GaussianStateEvolution(F=..., cov=...)`` which
        returns $\\mathcal{N}(F(x, u, t, t'), Q)$ -- the same pattern used
        throughout dynestyx for discrete-time Gaussian transitions.
        """
        mass, length, g_val, b_val, dt = (
            self.mass,
            self.length,
            self.g,
            self.b,
            self.dt,
        )
        max_torque = self.max_torque

        def transition_fn(x, u, t_now, t_next):
            theta, dtheta = x[0], x[1]
            u_val = (
                jnp.zeros(())
                if u is None
                else jnp.clip(u.squeeze(), -max_torque, max_torque)
            )
            ddtheta = (
                u_val - b_val * dtheta + mass * g_val * length * jnp.sin(theta)
            ) / (mass * length**2)
            new_dtheta = dtheta + ddtheta * dt
            new_theta = theta + new_dtheta * dt
            return jnp.array([new_theta, new_dtheta])

        return DynamicalModel(
            initial_condition=self._make_initial_condition(x0),
            state_evolution=GaussianStateEvolution(
                F=transition_fn,
                cov=process_noise**2 * jnp.eye(self.state_dim),
            ),
            observation_model=DiracIdentityObservation(),
            control_dim=1,
        )

    def to_dynamical_model(self, **kwargs) -> DynamicalModel:
        return self.to_continuous_dynamical_model(**kwargs)
