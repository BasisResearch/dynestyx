"""Policy / controller implementations for PILCO.

Provides linear and RBF controllers with analytic moment matching for
propagating Gaussian state uncertainty through the policy.

References:
    Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A Model-Based and
    Data-Efficient Approach to Policy Search. ICML, Eqs. 31-32.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


class LinearController(eqx.Module):
    """Linear controller: u = W @ x + b.

    For Gaussian inputs N(m, s), the output distribution is:
        mu_u = W @ m + b
        Sigma_u = W @ s @ W^T
        cov(x, u) = s @ W^T

    Attributes:
        W: Weight matrix, shape (control_dim, state_dim).
        b: Bias vector, shape (control_dim,).
    """

    W: Array
    b: Array

    def __init__(self, state_dim: int, control_dim: int, *, key: Array):
        self.W = 0.1 * jax.random.normal(key, (control_dim, state_dim))
        self.b = jnp.zeros(control_dim)

    def __call__(self, x: Array) -> Array:
        """Deterministic action for a known state."""
        return self.W @ x + self.b

    def compute_action(self, m: Array, s: Array) -> tuple[Array, Array, Array]:
        """Compute action distribution for uncertain state N(m, s).

        Args:
            m: State mean, shape (state_dim,).
            s: State covariance, shape (state_dim, state_dim).

        Returns:
            m_u: Action mean, shape (control_dim,).
            s_u: Action covariance, shape (control_dim, control_dim).
            c_xu: State-action cross-covariance, shape (state_dim, control_dim).
        """
        m_u = self.W @ m + self.b
        s_u = self.W @ s @ self.W.T
        c_xu = s @ self.W.T
        return m_u, s_u, c_xu


class RBFController(eqx.Module):
    """RBF network controller (Eq. 31-32).

    Uses Gaussian basis functions with analytic moment matching through
    the GP prediction machinery. Implemented as a deterministic GP
    (zero noise) where the "training points" are the basis function
    centers and the "targets" are the weights.

    Attributes:
        centers: RBF centers, shape (n_basis, state_dim).
        weights: RBF weights, shape (n_basis, control_dim).
        log_lengthscales: Log length-scales, shape (state_dim,).
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
        """Deterministic action for a known state."""
        diff = self.centers - x[None, :]  # (n_basis, state_dim)
        ls = self.lengthscales
        sq_dist = jnp.sum((diff / ls[None, :]) ** 2, axis=-1)  # (n_basis,)
        phi = jnp.exp(-0.5 * sq_dist)  # (n_basis,)
        return phi @ self.weights  # (control_dim,)

    def compute_action(self, m: Array, s: Array) -> tuple[Array, Array, Array]:
        """Compute action distribution for uncertain state N(m, s).

        Analytic moment matching through the RBF network. This mirrors
        the GP moment matching with a deterministic GP (iK=0).

        Args:
            m: State mean, shape (state_dim,).
            s: State covariance, shape (state_dim, state_dim).

        Returns:
            m_u: Action mean, shape (control_dim,).
            s_u: Action covariance, shape (control_dim, control_dim).
            c_xu: State-action cross-covariance, shape (state_dim, control_dim).
        """
        state_dim = m.shape[0]
        control_dim = self.weights.shape[1]

        Lambda = jnp.diag(self.lengthscales**2)
        Lambda_inv = jnp.diag(1.0 / self.lengthscales**2)
        nu = self.centers - m[None, :]  # (n_basis, state_dim)

        # === Mean: E[phi_i(x)] for uncertain x ~ N(m, s) ===
        sL = s + Lambda
        sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(state_dim))
        det_factor = jnp.linalg.det(sL) / jnp.prod(self.lengthscales**2)
        quad = jnp.sum(nu @ sL_inv * nu, axis=-1)  # (n_basis,)
        q = jnp.exp(-0.5 * quad) / jnp.sqrt(
            jnp.maximum(det_factor, 1e-12)
        )  # (n_basis,)

        m_u = q @ self.weights  # (control_dim,)

        # === Covariance: cov[u_a, u_b] ===
        R = s @ (2.0 * Lambda_inv) + jnp.eye(state_dim)
        R_inv = jnp.linalg.inv(R + 1e-8 * jnp.eye(state_dim))
        det_R = jnp.linalg.det(R)

        nu_L = nu @ Lambda_inv  # (n_basis, state_dim)
        R_inv_s = R_inv @ s
        t = nu_L @ R_inv_s  # (n_basis, state_dim)

        k_m = jnp.exp(-0.5 * jnp.sum(nu * nu_L, axis=-1))  # (n_basis,)

        q_diag = jnp.sum(nu_L * t, axis=-1)  # (n_basis,)
        q_cross = nu_L @ t.T  # (n_basis, n_basis)

        Q = (
            k_m[:, None]
            * k_m[None, :]
            / jnp.sqrt(jnp.maximum(det_R, 1e-12))
            * jnp.exp(0.5 * (q_diag[:, None] + q_diag[None, :] + 2.0 * q_cross))
        )  # (n_basis, n_basis)

        s_u = self.weights.T @ Q @ self.weights - jnp.outer(m_u, m_u)

        # === Cross-covariance: cov[x, u] ===
        c_xu = jnp.zeros((state_dim, control_dim))
        for d in range(control_dim):
            w_q_nu = (self.weights[:, d] * q)[:, None] * nu  # (n_basis, state_dim)
            c_xu = c_xu.at[:, d].set(s @ sL_inv @ jnp.sum(w_q_nu, axis=0))

        return m_u, s_u, c_xu


def squash_sin(m: Array, s: Array, max_action: Array) -> tuple[Array, Array, Array]:
    """Squash action through sin() for bounded controls.

    Analytically computes moments of max_action * sin(u) where u ~ N(m, s).

    Args:
        m: Action mean, shape (control_dim,).
        s: Action covariance, shape (control_dim, control_dim).
        max_action: Maximum action magnitude, shape (control_dim,).

    Returns:
        m_out: Squashed mean, shape (control_dim,).
        s_out: Squashed covariance, shape (control_dim, control_dim).
        c_out: Input-output cross-covariance, shape (control_dim, control_dim).
    """
    s_diag = jnp.diag(s)

    # E[sin(u)] = sin(m) * exp(-diag(s)/2)
    e = jnp.exp(-0.5 * s_diag)
    m_out = max_action * e * jnp.sin(m)

    # cov[sin(u_i), sin(u_j)]
    # Using the identity for E[sin(u_i)sin(u_j)] and E[cos(u_i)cos(u_j)]
    lq = -(s_diag[:, None] + s_diag[None, :]) / 2.0

    s_out = (
        jnp.exp(lq + s) * jnp.cos(m[:, None] - m[None, :])
        - jnp.exp(lq - s) * jnp.cos(m[:, None] + m[None, :])
    ) / 2.0
    s_out = max_action[:, None] * max_action[None, :] * s_out
    s_out = s_out - jnp.outer(m_out, m_out)

    # Cross-covariance: cov[u, sin(u)] = diag(max_action * cos(m) * exp(-diag(s)/2))
    c_out = jnp.diag(max_action * jnp.cos(m) * e)

    return m_out, s_out, c_out
