"""BlackJAX implementations for filter-based posterior inference."""

from collections.abc import Callable

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from blackjax.types import ArrayTree
from jax.flatten_util import ravel_pytree
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model

from dynestyx.inference.mcmc_configs import (
    BaseMCMCConfig,
    HMCConfig,
    MALAConfig,
    NUTSConfig,
    SGLDConfig,
)


def _has_chain_axis(initial_positions, num_chains: int) -> bool:
    leaves = jax.tree_util.tree_leaves(initial_positions)
    return (
        len(leaves) > 0
        and hasattr(leaves[0], "shape")
        and len(leaves[0].shape) >= 1
        and leaves[0].shape[0] == num_chains
    )


def _run_scan_kernel(rng_key, kernel, initial_state, num_steps):
    @jax.jit
    def one_step(state, key):
        state, _ = kernel(key, state)
        return state, state

    keys = jr.split(rng_key, num_steps)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    return states


def _run_scan_multiple_chains(chain_keys, kernel, initial_states, num_steps):
    if chain_keys.shape[0] == 1:
        single_states = _run_scan_kernel(
            chain_keys[0],
            kernel,
            jax.tree_util.tree_map(lambda x: x[0], initial_states),
            num_steps,
        )
        return jax.tree_util.tree_map(lambda x: x[None, ...], single_states)

    run_many = jax.pmap(
        _run_scan_kernel,
        in_axes=(0, None, 0, None),
        static_broadcasted_argnums=(1, 3),
    )
    return run_many(chain_keys, kernel, initial_states, num_steps)


def _run_blackjax(
    mcmc_key: jnp.ndarray,
    algorithm: blackjax.base.SamplingAlgorithm,
    initial_positions: ArrayTree,
    has_chain_axis: bool,
    num_chains: int,
    num_steps: int,
    transform_fn: Callable,
    num_warmup: int = 0,
    init_state_keys: jnp.ndarray | None = None,
) -> dict:
    if init_state_keys is None:
        initial_states = (
            jax.vmap(algorithm.init, in_axes=(0,))(initial_positions)  # type: ignore[call-arg]
            if has_chain_axis
            else algorithm.init(initial_positions)  # type: ignore[call-arg]
        )
    else:
        initial_states = (
            jax.vmap(algorithm.init, in_axes=(0, 0))(initial_positions, init_state_keys)
            if has_chain_axis
            else algorithm.init(initial_positions, init_state_keys[0])
        )

    chain_keys = jr.split(mcmc_key, num_chains)
    full_states = _run_scan_multiple_chains(
        chain_keys, algorithm.step, initial_states, num_steps
    )
    constrained = jax.jit(jax.vmap(jax.vmap(transform_fn)))(full_states.position)

    if num_warmup == 0:
        return constrained

    def _remove_warmup(samples):
        return {k: v[num_warmup:] for k, v in samples.items()}

    return jax.vmap(_remove_warmup)(constrained)


def run_blackjax_mcmc(
    mcmc_config: BaseMCMCConfig,
    rng_key: jnp.ndarray,
    model: Callable,
    obs_times: jnp.ndarray,
    obs_values: jnp.ndarray,
    ctrl_times: jnp.ndarray | None = None,
    ctrl_values: jnp.ndarray | None = None,
    *model_args,
    **model_kwargs,
) -> dict:
    """Run BlackJAX-based inference and return posterior samples."""
    rng_key, init_key_master = jr.split(rng_key)
    init_keys = jr.split(init_key_master, mcmc_config.num_chains)

    init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        rng_key=init_keys,
        model=model,
        model_args=(obs_times, obs_values, ctrl_times, ctrl_values, *model_args),
        model_kwargs=model_kwargs,
        dynamic_args=True,
        init_strategy=init_to_median,
    )
    initial_positions = init_params.z
    has_chain_axis = _has_chain_axis(initial_positions, mcmc_config.num_chains)

    logdensity_fn = lambda position: -potential_fn_gen(obs_times, obs_values)(position)
    transform_fn = postprocess_fn(obs_times, obs_values)

    if isinstance(mcmc_config, NUTSConfig):
        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)
        warmup_position = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )
        rng_key, warmup_key, mcmc_key = jr.split(rng_key, 3)
        ((_, warmup_parameters), _) = warmup.run(  # type: ignore
            warmup_key, warmup_position, num_steps=mcmc_config.num_warmup
        )
        nuts = blackjax.nuts(logdensity_fn, **warmup_parameters)
        return _run_blackjax(
            mcmc_key=mcmc_key,
            algorithm=nuts,
            initial_positions=initial_positions,
            has_chain_axis=has_chain_axis,
            num_chains=mcmc_config.num_chains,
            num_steps=mcmc_config.num_samples,
            transform_fn=transform_fn,
        )

    if isinstance(mcmc_config, HMCConfig):
        metric_position = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )
        flat_position, _ = ravel_pytree(metric_position)
        inv_mass_matrix = jnp.eye(flat_position.shape[0])
        hmc = blackjax.hmc(
            logdensity_fn,
            mcmc_config.step_size,
            inv_mass_matrix,
            mcmc_config.num_steps,
        )
        rng_key, mcmc_key = jr.split(rng_key)
        return _run_blackjax(
            mcmc_key=mcmc_key,
            algorithm=hmc,
            initial_positions=initial_positions,
            has_chain_axis=has_chain_axis,
            num_chains=mcmc_config.num_chains,
            num_steps=mcmc_config.num_samples + mcmc_config.num_warmup,
            transform_fn=transform_fn,
            num_warmup=mcmc_config.num_warmup,
        )

    if isinstance(mcmc_config, SGLDConfig):

        def grad_estimator(position, _):
            return jax.grad(logdensity_fn)(position)

        sgld = blackjax.sgld(grad_estimator)
        initial_positions = (
            initial_positions
            if has_chain_axis
            else jax.tree_util.tree_map(lambda x: x[None, ...], initial_positions)
        )

        def _run_chain(chain_key, init_position):
            chain_state = sgld.init(init_position)
            total_steps = mcmc_config.num_warmup + mcmc_config.num_samples
            step_ids = jnp.arange(1, total_steps + 1, dtype=jnp.float32)
            step_sizes = mcmc_config.step_size * step_ids ** (
                -mcmc_config.schedule_power
            )
            step_keys = jr.split(chain_key, total_steps)

            def _one_step(position, inputs):
                key_t, step_size_t = inputs
                next_position = sgld.step(key_t, position, None, step_size_t)
                return next_position, next_position

            _, chain_positions = jax.lax.scan(
                _one_step, chain_state, (step_keys, step_sizes)
            )
            post_warmup = jax.tree_util.tree_map(
                lambda x: x[mcmc_config.num_warmup :], chain_positions
            )
            return jax.vmap(transform_fn)(post_warmup)

        rng_key, mcmc_key = jr.split(rng_key)
        chain_keys = jr.split(mcmc_key, mcmc_config.num_chains)
        return jax.vmap(_run_chain)(chain_keys, initial_positions)

    if isinstance(mcmc_config, MALAConfig):
        mala = blackjax.mala(logdensity_fn, step_size=mcmc_config.step_size)
        rng_key, mcmc_key = jr.split(rng_key)
        return _run_blackjax(
            mcmc_key=mcmc_key,
            algorithm=mala,
            initial_positions=initial_positions,
            has_chain_axis=has_chain_axis,
            num_chains=mcmc_config.num_chains,
            num_steps=mcmc_config.num_samples + mcmc_config.num_warmup,
            transform_fn=transform_fn,
            num_warmup=mcmc_config.num_warmup,
        )

    raise ValueError(f"Invalid MCMC config: {mcmc_config}")
