"""Policy/controller implementations with analytic moment matching."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


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
        """Analytic moment matching through the RBF network."""
        state_dim = m.shape[0]
        control_dim = self.weights.shape[1]

        Lambda = jnp.diag(self.lengthscales**2)
        Lambda_inv = jnp.diag(1.0 / self.lengthscales**2)
        nu = self.centers - m[None, :]

        # Mean
        sL = s + Lambda
        sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(state_dim))
        det_factor = jnp.linalg.det(sL) / jnp.prod(self.lengthscales**2)
        quad = jnp.sum(nu @ sL_inv * nu, axis=-1)
        q = jnp.exp(-0.5 * quad) / jnp.sqrt(jnp.maximum(det_factor, 1e-12))

        m_u = q @ self.weights

        # Covariance
        R = s @ (2.0 * Lambda_inv) + jnp.eye(state_dim)
        R_inv = jnp.linalg.inv(R + 1e-8 * jnp.eye(state_dim))
        det_R = jnp.linalg.det(R)

        nu_L = nu @ Lambda_inv
        R_inv_s = R_inv @ s
        t = nu_L @ R_inv_s
        k_m = jnp.exp(-0.5 * jnp.sum(nu * nu_L, axis=-1))
        q_diag = jnp.sum(nu_L * t, axis=-1)
        q_cross = nu_L @ t.T

        Q = (
            k_m[:, None]
            * k_m[None, :]
            / jnp.sqrt(jnp.maximum(det_R, 1e-12))
            * jnp.exp(0.5 * (q_diag[:, None] + q_diag[None, :] + 2.0 * q_cross))
        )

        s_u = self.weights.T @ Q @ self.weights - jnp.outer(m_u, m_u)

        # Cross-covariance
        c_xu = jnp.zeros((state_dim, control_dim))
        for d in range(control_dim):
            w_q_nu = (self.weights[:, d] * q)[:, None] * nu
            c_xu = c_xu.at[:, d].set(s @ sL_inv @ jnp.sum(w_q_nu, axis=0))

        return m_u, s_u, c_xu


def squash_sin(m: Array, s: Array, max_action: Array) -> tuple[Array, Array, Array]:
    """Analytic moments of $u_{\\text{max}} \\sin(u)$ for $u \\sim \\mathcal{N}(m, s)$."""
    s_diag = jnp.diag(s)

    e = jnp.exp(-0.5 * s_diag)
    m_out = max_action * e * jnp.sin(m)

    lq = -(s_diag[:, None] + s_diag[None, :]) / 2.0

    s_out = (
        jnp.exp(lq + s) * jnp.cos(m[:, None] - m[None, :])
        - jnp.exp(lq - s) * jnp.cos(m[:, None] + m[None, :])
    ) / 2.0
    s_out = max_action[:, None] * max_action[None, :] * s_out
    s_out = s_out - jnp.outer(m_out, m_out)

    c_out = jnp.diag(max_action * jnp.cos(m) * e)

    return m_out, s_out, c_out
