"""Shared moment matching primitives for SE-ARD GPs (Deisenroth & Rasmussen, 2011)."""

import jax
import jax.numpy as jnp
from jax import Array

_JITTER = 1e-8
_EPS = 1e-12


def _inv(A: Array) -> Array:
    return jnp.linalg.inv(A + _JITTER * jnp.eye(A.shape[0]))


def compute_mean_and_q(
    nu: Array, s: Array, ls_sq: Array, sig_var: Array | float
) -> Array:
    """Expected kernel evaluations $q$ for uncertain input (Eq. 15)."""
    sL = s + jnp.diag(ls_sq)
    sL_inv = _inv(sL)
    det_factor = jnp.linalg.det(sL) / jnp.prod(ls_sq)
    quad = jnp.sum(nu @ sL_inv * nu, axis=-1)
    return sig_var / jnp.sqrt(jnp.maximum(det_factor, _EPS)) * jnp.exp(-0.5 * quad)


def compute_Q_matrix(
    nu: Array,
    s: Array,
    ls_sq_a: Array,
    ls_sq_b: Array,
    sv_a: Array | float,
    sv_b: Array | float,
) -> Array:
    """$Q$ matrix for covariance prediction (Eq. 22)."""
    d = nu.shape[1]
    La_inv = jnp.diag(1.0 / ls_sq_a)
    Lb_inv = jnp.diag(1.0 / ls_sq_b)

    R = s @ (La_inv + Lb_inv) + jnp.eye(d)
    R_inv = _inv(R)

    nu_La = nu @ La_inv
    nu_Lb = nu @ Lb_inv
    k_a = sv_a * jnp.exp(-0.5 * jnp.sum(nu * nu_La, axis=-1))
    k_b = sv_b * jnp.exp(-0.5 * jnp.sum(nu * nu_Lb, axis=-1))

    R_inv_s = R_inv @ s
    t_a, t_b = nu_La @ R_inv_s, nu_Lb @ R_inv_s
    q_aa = jnp.sum(nu_La * t_a, axis=-1)
    q_bb = jnp.sum(nu_Lb * t_b, axis=-1)
    q_ab = nu_La @ t_b.T

    return (
        k_a[:, None]
        * k_b[None, :]
        / jnp.sqrt(jnp.maximum(jnp.linalg.det(R), _EPS))
        * jnp.exp(0.5 * (q_aa[:, None] + q_bb[None, :] + 2.0 * q_ab))
    )


def compute_cross_covariance(
    nu: Array, s: Array, ls_sq: Array, beta: Array, q: Array
) -> Array:
    """Input-output cross-covariance for one output dimension."""
    sL_inv = _inv(s + jnp.diag(ls_sq))
    return s @ sL_inv @ jnp.sum((beta * q)[:, None] * nu, axis=0)


def compute_predictive_moments(
    nu: Array,
    s: Array,
    lengthscales: Array,
    signal_variance: Array,
    betas: list[Array],
    iKs: list[Array],
) -> tuple[Array, Array, Array]:
    """Full multi-output moment matching (Eqs. 14-23).

    Orchestrates mean, covariance, and cross-covariance computation
    across all $D$ output dimensions.

    Args:
        nu: Centered training inputs $\\tilde{x}_i - \\tilde{\\mu}$, shape ``(n, D+F)``.
        s: Input covariance, shape ``(D+F, D+F)``.
        lengthscales: Per-output lengthscales, shape ``(D, D+F)``.
        signal_variance: Per-output signal variances, shape ``(D,)``.
        betas: List of $D$ weight vectors, each shape ``(n,)``.
        iKs: List of $D$ inverse kernel matrices, each shape ``(n, n)``.

    Returns:
        M: Predictive mean, shape ``(D,)``.
        S: Predictive covariance, shape ``(D, D)``.
        V: Input-output cross-covariance, shape ``(D+F, D)``.
    """
    D = len(betas)
    input_dim = nu.shape[1]

    # Mean + q vectors
    qs = []
    M = jnp.zeros(D)
    for a in range(D):
        q = compute_mean_and_q(nu, s, lengthscales[a] ** 2, signal_variance[a])
        qs.append(q)
        M = M.at[a].set(betas[a] @ q)

    # Covariance (upper triangle, then symmetrize)
    S = jnp.zeros((D, D))
    for a in range(D):
        for b in range(a, D):
            Q = compute_Q_matrix(
                nu,
                s,
                lengthscales[a] ** 2,
                lengthscales[b] ** 2,
                signal_variance[a],
                signal_variance[b],
            )
            S_ab = betas[a] @ Q @ betas[b] - M[a] * M[b]
            if a == b:
                S_ab += signal_variance[a] - jnp.trace(iKs[a] @ Q)
            S = S.at[a, b].set(S_ab)
            if a != b:
                S = S.at[b, a].set(S_ab)
    S = S + 1e-6 * jnp.eye(D)

    # Cross-covariance
    V = jnp.zeros((input_dim, D))
    for a in range(D):
        V = V.at[:, a].set(
            compute_cross_covariance(nu, s, lengthscales[a] ** 2, betas[a], qs[a])
        )

    return M, S, V


def gp_log_marginal_likelihood(
    K_fn, X: Array, Y: Array, noise_variance: Array
) -> Array:
    """GP log marginal likelihood summed over $D$ independent outputs.

    Args:
        K_fn: Callable ``(X1, X2, a) -> kernel matrix``.
        X: Training inputs, shape ``(n, D+F)``.
        Y: Training targets, shape ``(n, D)``.
        noise_variance: Per-output noise variance, shape ``(D,)``.
    """
    n, D = Y.shape
    total: Array = jnp.array(0.0)
    for a in range(D):
        K = K_fn(X, X, a)
        L = jnp.linalg.cholesky(K + noise_variance[a] * jnp.eye(n) + 1e-6 * jnp.eye(n))
        alpha = jax.scipy.linalg.cho_solve((L, True), Y[:, a])
        total = total + (
            -0.5 * Y[:, a] @ alpha
            - jnp.sum(jnp.log(jnp.diag(L)))
            - 0.5 * n * jnp.log(2.0 * jnp.pi)
        )
    return total
