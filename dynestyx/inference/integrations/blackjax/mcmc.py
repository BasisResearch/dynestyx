"""BlackJAX implementations for filter-based posterior inference."""

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model

from dynestyx.inference.mcmc_configs import HMCConfig, NUTSConfig, SGLDConfig


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
    run_many = jax.pmap(
        _run_scan_kernel,
        in_axes=(0, None, 0, None),
        static_broadcasted_argnums=(1, 3),
    )
    return run_many(chain_keys, kernel, initial_states, num_steps)


def run_blackjax_mcmc(
    mcmc_config, rng_key, data_conditioned_model, obs_times, obs_values
):
    """Run BlackJAX-based inference (`NUTS`, `HMC`, or `SGLD`) and return samples."""
    rng_key, init_key_master = jr.split(rng_key)
    init_keys = jr.split(init_key_master, mcmc_config.num_chains)

    init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        rng_key=init_keys,
        model=data_conditioned_model,
        model_args=(obs_times, obs_values),
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
        ((_, warmup_parameters), _) = warmup.run(
            warmup_key, warmup_position, num_steps=mcmc_config.num_warmup
        )
        nuts = blackjax.nuts(logdensity_fn, **warmup_parameters)
        initial_states = (
            jax.vmap(nuts.init, in_axes=(0,))(initial_positions)
            if has_chain_axis
            else nuts.init(initial_positions)
        )
        chain_keys = jr.split(mcmc_key, mcmc_config.num_chains)
        full_states = _run_scan_multiple_chains(
            chain_keys, nuts.step, initial_states, mcmc_config.num_samples
        )
        return jax.jit(jax.vmap(jax.vmap(transform_fn)))(full_states.position)

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
        initial_states = (
            jax.vmap(hmc.init, in_axes=(0,))(initial_positions)
            if has_chain_axis
            else hmc.init(initial_positions)
        )
        rng_key, mcmc_key = jr.split(rng_key)
        chain_keys = jr.split(mcmc_key, mcmc_config.num_chains)
        full_states = _run_scan_multiple_chains(
            chain_keys,
            hmc.step,
            initial_states,
            mcmc_config.num_samples + mcmc_config.num_warmup,
        )
        constrained = jax.jit(jax.vmap(jax.vmap(transform_fn)))(full_states.position)

        def _remove_warmup(samples):
            return {k: v[mcmc_config.num_warmup :] for k, v in samples.items()}

        return jax.vmap(_remove_warmup)(constrained)

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

    raise ValueError(f"Invalid MCMC config: {mcmc_config}")
