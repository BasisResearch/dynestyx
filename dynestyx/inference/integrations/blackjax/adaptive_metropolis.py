"""Adaptive random-walk Metropolis for the BlackJAX integration."""

# The diminishing adaptation rule is inspired by PFJAX:
# https://github.com/mlysy/pfjax/blob/97652aa1bdff73a92c0286549b010e99cc6f7264/src/pfjax/mcmc.py

from collections.abc import Callable
from typing import NamedTuple

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.base import SamplingAlgorithm, build_sampling_algorithm


def adapt_proposal_scale(
    proposal_scale: jax.Array,
    acceptance_rate: jax.Array,
    n_iter: jax.Array,
    *,
    target_acceptance_rate: float,
    adaptation_rate: float,
    max_adaptation: float,
) -> jax.Array:
    """Apply PFJAX's diminishing proposal-scale adaptation rule."""
    delta = jnp.minimum(n_iter**-adaptation_rate, max_adaptation)
    return jnp.exp(
        jnp.log(proposal_scale)
        - delta * jnp.sign(target_acceptance_rate - acceptance_rate)
    )


def resolve_proposal_scale(
    initial_proposal_scale: jax.Array,
    latent_size: int,
) -> jax.Array:
    """Broadcast a scalar scale or validate a coordinatewise scale vector."""
    proposal_scale = jnp.asarray(initial_proposal_scale)
    if proposal_scale.ndim == 0:
        return jnp.full((latent_size,), proposal_scale)
    if proposal_scale.shape != (latent_size,):
        raise ValueError(
            "initial_proposal_scale must be scalar or have one value per "
            f"flattened unconstrained coordinate; expected {(latent_size,)}, "
            f"got {proposal_scale.shape}"
        )
    return proposal_scale


class AdaptiveMetropolisState(NamedTuple):
    """State carried by the adaptive Metropolis kernel."""

    position: jax.Array
    proposal_scale: jax.Array
    n_accept: jax.Array
    n_iter: jax.Array


class AdaptiveMetropolisInfo(NamedTuple):
    """Per-coordinate acceptance indicators for one transition."""

    is_accepted: jax.Array


def init(
    position: jax.Array,
    logdensity_fn: Callable,
    proposal_scale: jax.Array,
) -> AdaptiveMetropolisState:
    """Initialize the kernel state."""
    del logdensity_fn
    return AdaptiveMetropolisState(
        position,
        proposal_scale,
        jnp.zeros_like(position),
        jnp.array(0.0),
    )


def build_kernel() -> Callable:
    """Build one complete componentwise random-walk Metropolis transition."""
    rmh_step = blackjax.mcmc.random_walk.build_rmh()

    def kernel(
        rng_key: jax.Array,
        state: AdaptiveMetropolisState,
        logdensity_fn: Callable,
        target_acceptance_rate: float,
        adaptation_rate: float,
        max_adaptation: float,
        num_warmup: int,
    ) -> tuple[AdaptiveMetropolisState, AdaptiveMetropolisInfo]:
        rw_state = blackjax.mcmc.random_walk.init(state.position, logdensity_fn)

        def update_coordinate(rw_state, coordinate_and_key):
            coordinate, coordinate_key = coordinate_and_key

            def propose(key, position):
                jump = state.proposal_scale[coordinate] * jr.normal(
                    key, dtype=position.dtype
                )
                return position.at[coordinate].add(jump)

            rw_state, info = rmh_step(
                coordinate_key,
                rw_state,
                logdensity_fn,
                propose,
            )
            return rw_state, info.is_accepted

        rw_state, is_accepted = jax.lax.scan(
            update_coordinate,
            rw_state,
            (jnp.arange(state.position.size), jr.split(rng_key, state.position.size)),
        )
        n_iter = state.n_iter + 1.0
        n_accept = state.n_accept + is_accepted
        adapted_scale = adapt_proposal_scale(
            state.proposal_scale,
            n_accept / n_iter,
            n_iter,
            target_acceptance_rate=target_acceptance_rate,
            adaptation_rate=adaptation_rate,
            max_adaptation=max_adaptation,
        )
        proposal_scale = jnp.where(
            state.n_iter < num_warmup,
            adapted_scale,
            state.proposal_scale,
        )
        next_state = AdaptiveMetropolisState(
            rw_state.position,
            proposal_scale,
            n_accept,
            n_iter,
        )
        return next_state, AdaptiveMetropolisInfo(is_accepted)

    return kernel


def adaptive_metropolis(
    logdensity_fn: Callable,
    proposal_scale: jax.Array,
    *,
    target_acceptance_rate: float,
    adaptation_rate: float,
    max_adaptation: float,
    num_warmup: int,
) -> SamplingAlgorithm:
    """Return the adaptive Metropolis kernel as a BlackJAX algorithm."""
    return build_sampling_algorithm(
        build_kernel(),
        init,
        logdensity_fn,
        init_args=(proposal_scale,),
        kernel_args=(
            target_acceptance_rate,
            adaptation_rate,
            max_adaptation,
            num_warmup,
        ),
    )
