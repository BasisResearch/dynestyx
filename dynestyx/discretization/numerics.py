"""Numerical safeguards shared by Gaussian SDE discretizers."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Real


def _positive_interval(t_now, t_next) -> Array:
    """Return a validated positive interval length."""
    h = jnp.asarray(t_next) - jnp.asarray(t_now)
    return eqx.error_if(h, h <= 0, "Discretization requires t_next > t_now.")


def _symmetrize(cov: Real[Array, "... state_dim state_dim"]) -> Array:
    return 0.5 * (cov + jnp.swapaxes(cov, -1, -2))


def _finalize_covariance(
    cov: Real[Array, "... state_dim state_dim"],
    *,
    covariance_jitter: float,
) -> Real[Array, "... state_dim state_dim"]:
    """Symmetrize, add explicit jitter, and require positive definiteness."""
    cov = _symmetrize(cov)
    state_dim = cov.shape[-1]
    cov = cov + covariance_jitter * jnp.eye(state_dim, dtype=cov.dtype)
    cov = eqx.error_if(
        cov,
        ~jnp.all(jnp.isfinite(cov)),
        "Discretization produced a non-finite transition covariance.",
    )
    validation_cholesky = jnp.linalg.cholesky(jax.lax.stop_gradient(cov))
    cholesky_is_valid = jnp.all(jnp.isfinite(validation_cholesky)) & jnp.all(
        jnp.diagonal(validation_cholesky, axis1=-2, axis2=-1) > 0
    )
    return eqx.error_if(
        cov,
        ~cholesky_is_valid,
        "Transition covariance is singular or indefinite. Set a positive "
        "covariance_jitter explicitly if a nondegenerate Gaussian approximation "
        "is appropriate.",
    )
