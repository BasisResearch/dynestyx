"""Simple JAX-compatible environments for PILCO.

Provides pure-JAX environment implementations that can be used for data
collection in the PILCO loop without requiring external dependencies.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from dynestyx.models.core import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    ObservationModel,
)
from dynestyx.types import Control, State, Time


class InvertedPendulumEnv(eqx.Module):
    """Inverted pendulum (swing-up) environment.

    State: [theta, dtheta] where theta is the angle from upright.
    Action: torque in [-max_torque, max_torque].

    The dynamics are:
        dtheta/dt = dtheta
        ddtheta/dt = (u - b*dtheta + m*g*l*sin(theta)) / (m*l^2)

    Attributes:
        m: Pendulum mass.
        l: Pendulum length.
        g: Gravitational acceleration.
        b: Damping coefficient.
        dt: Integration time step.
        max_torque: Maximum torque.
    """

    mass: float = 1.0
    length: float = 1.0
    g: float = 9.81
    b: float = 0.0
    dt: float = 0.05
    max_torque: float = 2.0

    def step(self, state: Array, action: Array) -> Array:
        """Simulate one step of the pendulum dynamics.

        Uses Euler integration.

        Args:
            state: Current state [theta, dtheta], shape (2,).
            action: Torque, shape (1,) or scalar.

        Returns:
            Next state [theta, dtheta], shape (2,).
        """
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
        """Roll out a policy for T steps.

        Args:
            policy_fn: Callable mapping state -> action.
            x0: Initial state, shape (state_dim,).
            T: Number of steps.
            key: Optional PRNG key (unused, for API compatibility).

        Returns:
            states: States visited, shape (T+1, state_dim).
            actions: Actions taken, shape (T, control_dim).
            next_states: Next states, shape (T, state_dim).
        """

        def scan_fn(state, _):
            action = policy_fn(state)
            next_state = self.step(state, action)
            return next_state, (state, action, next_state)

        _, (states, actions, next_states) = jax.lax.scan(
            scan_fn, x0, None, length=T
        )
        # Append final state
        all_states = jnp.concatenate(
            [states, next_states[-1:]], axis=0
        )
        return all_states, actions, next_states

    def random_rollout(
        self, x0: Array, T: int, key: Array
    ) -> tuple[Array, Array, Array]:
        """Roll out with random actions for T steps.

        Args:
            x0: Initial state, shape (state_dim,).
            T: Number of steps.
            key: PRNG key for random actions.

        Returns:
            states, actions, next_states as in rollout().
        """
        keys = jax.random.split(key, T)

        def scan_fn(state, k):
            action = jax.random.uniform(
                k, (1,), minval=-self.max_torque, maxval=self.max_torque
            )
            next_state = self.step(state, action)
            return next_state, (state, action, next_state)

        _, (states, actions, next_states) = jax.lax.scan(
            scan_fn, x0, keys
        )
        all_states = jnp.concatenate(
            [states, next_states[-1:]], axis=0
        )
        return all_states, actions, next_states

    def to_dynamical_model(self) -> DynamicalModel:
        """Convert to a dynestyx DynamicalModel for use with effectful handlers.

        Creates a continuous-time DynamicalModel with the pendulum drift
        and identity observation model.
        """
        import numpyro.distributions as dist

        mass, length, g_val, b_val = self.mass, self.length, self.g, self.b

        def drift(x: State, u: Control, t: Time) -> State:
            theta, dtheta = x[0], x[1]
            u_val = jnp.zeros(()) if u is None else u.squeeze()
            ddtheta = (
                u_val - b_val * dtheta + mass * g_val * length * jnp.sin(theta)
            ) / (mass * length**2)
            return jnp.array([dtheta, ddtheta])

        state_evolution = ContinuousTimeStateEvolution(drift=drift)

        class IdentityObs(ObservationModel):
            noise_std: float = 0.01

            def __call__(self, x, u, t):
                return dist.MultivariateNormal(
                    loc=x, covariance_matrix=self.noise_std**2 * jnp.eye(2)
                )

        return DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(2), covariance_matrix=0.1 * jnp.eye(2)
            ),
            state_evolution=state_evolution,
            observation_model=IdentityObs(),
            control_dim=1,
        )
