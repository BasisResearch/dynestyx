"""Policy/controller implementations with analytic moment matching."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from dynestyx.pilco.moment_matching import (
    compute_cross_covariance,
    compute_mean_and_q,
    compute_Q_matrix,
)


class LinearController(eqx.Module):
    """
    Linear controller: $u = Wx + b$.

    For Gaussian inputs $\\mathcal{N}(m, s)$:
    $\\mu_u = Wm + b$, $\\Sigma_u = W s W^\\top$, $\\mathrm{cov}[x, u] = s W^\\top$.
    """

    W: Array
    b: Array

    def __init__(self, state_dim: int, control_dim: int, *, key: Array):
        self.W = 0.1 * jax.random.normal(key, (control_dim, state_dim))
        self.b = jnp.zeros(control_dim)

    def __call__(self, x: Array) -> Array:
        return self.W @ x + self.b

    def compute_action(self, m: Array, s: Array) -> tuple[Array, Array, Array]:
        m_u = self.W @ m + self.b
        s_u = self.W @ s @ self.W.T
        c_xu = s @ self.W.T
        return m_u, s_u, c_xu


class RBFController(eqx.Module):
    """
    RBF network controller (Eq. 31-32).

    Implemented as a deterministic GP whose "training points" are basis
    function centers and "targets" are weights.
    """

    centers: Array
    weights: Array
    log_lengthscales: Array

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        n_basis: int = 50,
        *,
        key: Array,
    ):
        k1, k2 = jax.random.split(key)
        self.centers = jax.random.normal(k1, (n_basis, state_dim))
        self.weights = 0.1 * jax.random.normal(k2, (n_basis, control_dim))
        self.log_lengthscales = jnp.zeros(state_dim)

    @property
    def lengthscales(self) -> Array:
        return jnp.exp(self.log_lengthscales)

    def __call__(self, x: Array) -> Array:
        diff = self.centers - x[None, :]
        ls = self.lengthscales
        sq_dist = jnp.sum((diff / ls[None, :]) ** 2, axis=-1)
        phi = jnp.exp(-0.5 * sq_dist)
        return phi @ self.weights

    def compute_action(self, m: Array, s: Array) -> tuple[Array, Array, Array]:
        """Analytic moment matching through the RBF network.

        Uses shared primitives from ``moment_matching`` -- the same math
        as ``MGPR.predict_uncertain`` but with signal_variance=1
        (deterministic GP, iK=0).
        """
        state_dim = m.shape[0]
        control_dim = self.weights.shape[1]
        ls_sq = self.lengthscales**2
        nu = self.centers - m[None, :]

        q = compute_mean_and_q(nu, s, ls_sq, 1.0)
        m_u = q @ self.weights

        Q = compute_Q_matrix(nu, s, ls_sq, ls_sq, 1.0, 1.0)
        s_u = self.weights.T @ Q @ self.weights - jnp.outer(m_u, m_u)

        c_xu = jnp.zeros((state_dim, control_dim))
        for d in range(control_dim):
            c_xu = c_xu.at[:, d].set(
                compute_cross_covariance(nu, s, ls_sq, self.weights[:, d], q)
            )

        return m_u, s_u, c_xu


Controller = LinearController | RBFController
