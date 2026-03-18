"""Simple JAX-compatible environments for PILCO.

Provides pure-JAX environment implementations that can be used for data
collection in the PILCO loop. Each environment can produce a dynestyx
``DynamicalModel`` (both continuous-time and discrete-time variants) for
use with effectful handlers.
"""

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
    """Inverted pendulum (swing-up) environment.

    State: $[\\theta, \\dot\\theta]$ where $\\theta$ is the angle from upright.
    Action: torque in $[-\\text{max\\_torque}, \\text{max\\_torque}]$.

    The dynamics are:

    $$\\ddot\\theta = \\frac{u - b\\dot\\theta + mgl\\sin(\\theta)}{ml^2}$$

    Provides integration with dynestyx:
    - ``to_continuous_dynamical_model()``: ContinuousTimeStateEvolution for use with
      ``ODESimulator``/``SDESimulator`` effectful handlers
    - ``to_discrete_dynamical_model()``: DiscreteTimeStateEvolution for use with
      ``DiscreteTimeSimulator``/``Filter`` effectful handlers
    """

    mass: float = 1.0
    length: float = 1.0
    g: float = 9.81
    b: float = 0.0
    dt: float = 0.05
    max_torque: float = 2.0

    def step(self, state: Array, action: Array) -> Array:
        """Simulate one step of the pendulum dynamics (Euler integration)."""
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
        """Roll out a policy for T steps."""

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
        """Roll out with random actions for T steps."""
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

    def _make_initial_condition(self, x0: Array | None = None):
        """Create initial condition distribution."""
        if x0 is not None:
            return dist.MultivariateNormal(loc=x0, covariance_matrix=0.01 * jnp.eye(2))
        return dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.1 * jnp.eye(2)
        )

    def to_continuous_dynamical_model(
        self, x0: Array | None = None, obs_noise: float = 0.01
    ) -> DynamicalModel:
        """Create a continuous-time dynestyx ``DynamicalModel``.

        For use with ``ODESimulator`` or ``SDESimulator`` effectful handlers
        via ``dsx.sample()``.

        Args:
            x0: Initial state mean (defaults to zeros).
            obs_noise: Observation noise std dev.
        """
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
                R=obs_noise**2 * jnp.eye(2),
            ),
            control_dim=1,
        )

    def to_discrete_dynamical_model(
        self, x0: Array | None = None, process_noise: float = 0.001
    ) -> DynamicalModel:
        """Create a discrete-time dynestyx ``DynamicalModel``.

        Uses Euler-integrated pendulum dynamics as a
        ``GaussianStateEvolution`` (discrete-time). For use with
        ``DiscreteTimeSimulator`` or ``Filter`` effectful handlers
        via ``dsx.sample()``.

        Args:
            x0: Initial state mean (defaults to zeros).
            process_noise: Process noise std dev for transition.
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
                cov=process_noise**2 * jnp.eye(2),
            ),
            observation_model=DiracIdentityObservation(),
            control_dim=1,
        )

    # Backwards-compatible alias
    def to_dynamical_model(self, **kwargs) -> DynamicalModel:
        """Alias for ``to_continuous_dynamical_model``."""
        return self.to_continuous_dynamical_model(**kwargs)
