"""Joint adaptive random-walk Metropolis for BlackJAX.

Warmup follows Andrieu & Thoms (2008), Algorithm 4:
https://people.eecs.berkeley.edu/~jordan/sail/readings/andrieu-thoms.pdf
"""

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm
from blackjax.mcmc.random_walk import RWState, build_rmh


class AdaptiveMetropolisState(NamedTuple):
    """Position, cached density, and warmup adaptation statistics."""

    position: jax.Array
    logdensity: jax.Array
    mean: jax.Array
    covariance: jax.Array
    log_multiplier: jax.Array
    n_iter: jax.Array


class AdaptiveMetropolisInfo(NamedTuple):
    """Acceptance indicator and probability for one joint proposal."""

    is_accepted: jax.Array
    acceptance_rate: jax.Array


def proposal_covariance(state: AdaptiveMetropolisState) -> jax.Array:
    """Return the symmetric, regularized covariance used for proposals."""
    covariance = jnp.exp(state.log_multiplier) * state.covariance
    covariance = (covariance + covariance.T) / 2
    dimension = state.position.size
    floor = jnp.maximum(
        jnp.asarray(1e-12, dtype=covariance.dtype),
        dimension
        * jnp.finfo(covariance.dtype).eps
        * jnp.max(jnp.abs(jnp.diag(covariance))),
    )
    return covariance + floor * jnp.eye(dimension, dtype=covariance.dtype)


def _finite_logdensity(logdensity_fn: Callable, position: jax.Array) -> jax.Array:
    value = logdensity_fn(position)
    return jnp.where(jnp.isfinite(value), value, -jnp.inf)


def init(
    position: jax.Array,
    logdensity_fn: Callable,
    proposal_scale: jax.Array,
) -> AdaptiveMetropolisState:
    """Initialize a flat chain and evaluate its log density once."""
    if position.ndim != 1 or position.size == 0:
        raise ValueError("position must be a nonempty flattened parameter vector")
    proposal_scale = jnp.broadcast_to(proposal_scale, position.shape).astype(
        position.dtype
    )
    return AdaptiveMetropolisState(
        position,
        _finite_logdensity(logdensity_fn, position),
        position,
        jnp.diag(proposal_scale**2),
        jnp.log(jnp.asarray(2.38**2 / position.size, dtype=position.dtype)),
        jnp.array(0, dtype=jnp.int32),
    )


def build_kernel() -> Callable:
    """Build one joint Metropolis transition with optional warmup adaptation."""
    rmh_step = build_rmh()

    def kernel(
        rng_key: jax.Array,
        state: AdaptiveMetropolisState,
        logdensity_fn: Callable,
        target_acceptance_rate: float,
        adaptation_rate: float,
        num_warmup: int,
    ) -> tuple[AdaptiveMetropolisState, AdaptiveMetropolisInfo]:
        chol = jnp.linalg.cholesky(proposal_covariance(state))

        def propose(key, position):
            return position + chol @ jr.normal(
                key, position.shape, dtype=position.dtype
            )

        rw_state, info = rmh_step(
            rng_key,
            RWState(state.position, state.logdensity),
            lambda position: _finite_logdensity(logdensity_fn, position),
            propose,
        )
        n_iter = state.n_iter + 1

        def adapt(_):
            gain = n_iter.astype(state.position.dtype) ** -adaptation_rate
            delta = rw_state.position - state.mean
            return (
                state.mean + gain * delta,
                state.covariance + gain * (jnp.outer(delta, delta) - state.covariance),
                state.log_multiplier
                + gain * (info.acceptance_rate - target_acceptance_rate),
            )

        mean, covariance, log_multiplier = jax.lax.cond(
            state.n_iter < num_warmup,
            adapt,
            lambda _: (state.mean, state.covariance, state.log_multiplier),
            operand=None,
        )
        return (
            AdaptiveMetropolisState(
                rw_state.position,
                rw_state.logdensity,
                mean,
                covariance,
                log_multiplier,
                n_iter,
            ),
            AdaptiveMetropolisInfo(info.is_accepted, info.acceptance_rate),
        )

    return kernel


def adaptive_metropolis(
    logdensity_fn: Callable,
    proposal_scale: jax.Array,
    *,
    target_acceptance_rate: float,
    adaptation_rate: float,
    num_warmup: int,
) -> SamplingAlgorithm:
    """Return joint adaptive Metropolis as a BlackJAX sampling algorithm."""
    return build_sampling_algorithm(
        build_kernel(),
        init,
        logdensity_fn,
        init_args=(proposal_scale,),
        kernel_args=(target_acceptance_rate, adaptation_rate, num_warmup),
    )
