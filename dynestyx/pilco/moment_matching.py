"""Shared moment matching primitives for MGPR and RBF controllers.

Implements the core analytic Gaussian moment matching through SE-ARD
kernels (Deisenroth & Rasmussen, 2011, Eqs. 14-23). Used by both
``MGPR.predict_given_factorizations`` and ``RBFController.compute_action``.
"""

import jax.numpy as jnp
from jax import Array


def compute_mean_and_q(
    nu: Array, s: Array, lengthscales_sq: Array, signal_variance: float
) -> Array:
    """
    Compute expected kernel evaluations $q$ for uncertain input (Eq. 15).

    Given centered training inputs $\\nu_i = \\tilde{x}_i - \\tilde{\\mu}$
    and input covariance $s$, computes:

    $$q_i = \\frac{\\alpha^2}{\\sqrt{|s \\Lambda^{-1} + I|}}
    \\exp\\!\\left(-\\frac{1}{2} \\nu_i^\\top (s + \\Lambda)^{-1} \\nu_i\\right)$$

    Args:
        nu: Centered training inputs, shape ``(n, D+F)``.
        s: Input covariance, shape ``(D+F, D+F)``.
        lengthscales_sq: Squared length-scales, shape ``(D+F,)``.
        signal_variance: Signal variance $\\alpha^2$.

    Returns:
        q: Expected kernel evaluations, shape ``(n,)``.
    """
    input_dim = nu.shape[1]
    Lambda = jnp.diag(lengthscales_sq)
    sL = s + Lambda
    sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(input_dim))
    det_factor = jnp.linalg.det(sL) / jnp.prod(lengthscales_sq)
    quad = jnp.sum(nu @ sL_inv * nu, axis=-1)
    return (
        signal_variance
        / jnp.sqrt(jnp.maximum(det_factor, 1e-12))
        * jnp.exp(-0.5 * quad)
    )


def compute_Q_matrix(
    nu: Array,
    s: Array,
    lengthscales_sq_a: Array,
    lengthscales_sq_b: Array,
    signal_variance_a: float,
    signal_variance_b: float,
) -> Array:
    """
    Compute the $Q$ matrix for covariance prediction (Eq. 22).

    $$Q_{ij} = \\frac{k_a(\\tilde{x}_i, \\tilde{\\mu})\\, k_b(\\tilde{x}_j,
    \\tilde{\\mu})}{\\sqrt{|R|}}\\, \\exp\\!\\left(\\frac{1}{2} z_{ij}^\\top
    R^{-1} \\tilde{\\Sigma}\\, z_{ij}\\right)$$

    Args:
        nu: Centered training inputs, shape ``(n, D+F)``.
        s: Input covariance, shape ``(D+F, D+F)``.
        lengthscales_sq_a: Squared length-scales for GP a, shape ``(D+F,)``.
        lengthscales_sq_b: Squared length-scales for GP b, shape ``(D+F,)``.
        signal_variance_a: Signal variance for GP a.
        signal_variance_b: Signal variance for GP b.

    Returns:
        Q: Matrix of shape ``(n, n)``.
    """
    input_dim = nu.shape[1]
    Lambda_a_inv = jnp.diag(1.0 / lengthscales_sq_a)
    Lambda_b_inv = jnp.diag(1.0 / lengthscales_sq_b)

    R = s @ (Lambda_a_inv + Lambda_b_inv) + jnp.eye(input_dim)
    R_inv = jnp.linalg.inv(R + 1e-8 * jnp.eye(input_dim))
    det_R = jnp.linalg.det(R)

    nu_La = nu @ Lambda_a_inv
    nu_Lb = nu @ Lambda_b_inv

    k_a_m = signal_variance_a * jnp.exp(-0.5 * jnp.sum(nu * nu_La, axis=-1))
    k_b_m = signal_variance_b * jnp.exp(-0.5 * jnp.sum(nu * nu_Lb, axis=-1))

    R_inv_s = R_inv @ s
    t_a = nu_La @ R_inv_s
    t_b = nu_Lb @ R_inv_s
    q_aa = jnp.sum(nu_La * t_a, axis=-1)
    q_bb = jnp.sum(nu_Lb * t_b, axis=-1)
    q_ab = nu_La @ t_b.T

    return (
        k_a_m[:, None]
        * k_b_m[None, :]
        / jnp.sqrt(jnp.maximum(det_R, 1e-12))
        * jnp.exp(0.5 * (q_aa[:, None] + q_bb[None, :] + 2.0 * q_ab))
    )


def compute_cross_covariance(
    nu: Array, s: Array, lengthscales_sq: Array, beta: Array, q: Array
) -> Array:
    """
    Compute input-output cross-covariance for one output dimension.

    $$\\mathrm{cov}[\\tilde{x}, \\Delta_a] = s (s + \\Lambda_a)^{-1}
    \\sum_i \\beta_{ai} q_{ai} \\nu_i$$

    Args:
        nu: Centered training inputs, shape ``(n, D+F)``.
        s: Input covariance, shape ``(D+F, D+F)``.
        lengthscales_sq: Squared length-scales, shape ``(D+F,)``.
        beta: GP weight vector, shape ``(n,)``.
        q: Expected kernel evaluations, shape ``(n,)``.

    Returns:
        Cross-covariance vector, shape ``(D+F,)``.
    """
    input_dim = nu.shape[1]
    Lambda = jnp.diag(lengthscales_sq)
    sL = s + Lambda
    sL_inv = jnp.linalg.inv(sL + 1e-8 * jnp.eye(input_dim))
    weighted_nu = (beta * q)[:, None] * nu
    return s @ sL_inv @ jnp.sum(weighted_nu, axis=0)
