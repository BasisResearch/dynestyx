"""Reward functions with analytic expected reward under Gaussian uncertainty."""

import equinox as eqx
import jax.numpy as jnp
from jax import Array


class ExponentialReward(eqx.Module):
    """
    Saturating exponential reward (negative of Eq. 25 cost).

    $$r(x) = \\exp\\!\\left(-\\frac{1}{2}(x - x_\\text{target})^\\top W
    (x - x_\\text{target})\\right)$$

    Expected reward under $\\mathcal{N}(\\mu, \\Sigma)$ computed analytically.
    """

    W: Array
    target: Array

    def __init__(self, state_dim: int, target: Array, W: Array | None = None):
        self.target = target
        self.W = W if W is not None else jnp.eye(state_dim)

    def __call__(self, m: Array, s: Array) -> tuple[Array, Array]:
        D = m.shape[0]
        SW = s @ self.W
        iSpW = jnp.linalg.solve(jnp.eye(D) + SW, self.W)

        diff = m - self.target
        mu_r = jnp.exp(-0.5 * diff @ iSpW @ diff) / jnp.sqrt(
            jnp.linalg.det(jnp.eye(D) + SW)
        )

        i2SpW = jnp.linalg.solve(jnp.eye(D) + 2.0 * SW, 2.0 * self.W)
        r2 = jnp.exp(-0.5 * diff @ i2SpW @ diff) / jnp.sqrt(
            jnp.linalg.det(jnp.eye(D) + 2.0 * SW)
        )
        s_r = r2 - mu_r**2

        return mu_r, s_r
