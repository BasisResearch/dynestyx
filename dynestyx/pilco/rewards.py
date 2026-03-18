"""Reward functions for PILCO with analytic expected reward under Gaussian uncertainty.

Implements the saturating cost function from Deisenroth & Rasmussen (2011), Eq. 25.
"""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class ExponentialReward(eqx.Module):
    """Saturating exponential reward (negative of Eq. 25 cost).

    For a target state x_target and width matrix W:
        reward(x) = exp(-0.5 * (x - x_target)^T @ W @ (x - x_target))

    The expected reward under a Gaussian state distribution p(x) = N(mu, Sigma)
    is computed analytically (Eq. 24).

    Attributes:
        W: Precision/weight matrix, shape (state_dim, state_dim).
        target: Target state, shape (state_dim,).
    """

    W: Array
    target: Array

    def __init__(self, state_dim: int, target: Array, W: Array | None = None):
        """Initialize ExponentialReward.

        Args:
            state_dim: Dimension of the state.
            target: Target state, shape (state_dim,).
            W: Weight/precision matrix. Defaults to identity.
        """
        self.target = target
        self.W = W if W is not None else jnp.eye(state_dim)

    def __call__(self, m: Array, s: Array) -> tuple[Array, Array]:
        """Compute expected reward and its variance under Gaussian state.

        Args:
            m: State mean, shape (state_dim,).
            s: State covariance, shape (state_dim, state_dim).

        Returns:
            mu_r: Expected reward (scalar).
            s_r: Reward variance (scalar).
        """
        D = m.shape[0]
        SW = s @ self.W
        iSpW = jnp.linalg.solve(jnp.eye(D) + SW, self.W)

        diff = m - self.target
        mu_r = jnp.exp(-0.5 * diff @ iSpW @ diff) / jnp.sqrt(
            jnp.linalg.det(jnp.eye(D) + SW)
        )

        # Second moment for variance
        i2SpW = jnp.linalg.solve(jnp.eye(D) + 2.0 * SW, 2.0 * self.W)
        r2 = jnp.exp(-0.5 * diff @ i2SpW @ diff) / jnp.sqrt(
            jnp.linalg.det(jnp.eye(D) + 2.0 * SW)
        )
        s_r = r2 - mu_r**2

        return mu_r, s_r
