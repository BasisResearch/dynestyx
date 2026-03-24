"""BlackJAX implementations for filter-based posterior inference."""

from collections.abc import Callable

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from numpyro import handlers
from numpyro.infer import init_to_median
from numpyro.infer.util import initialize_model, potential_energy

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


def _run_chain_scan(rng_key, make_step, initial_state, num_steps):
    """Scan ``num_steps`` MCMC steps, passing a fresh density key to each."""

    def one_step(state, keys):
        mcmc_key, density_key = keys
        state, _ = make_step(density_key)(mcmc_key, state)
        return state, state

    key_mcmc, key_density = jr.split(rng_key)
    _, states = jax.lax.scan(
        one_step,
        initial_state,
        (jr.split(key_mcmc, num_steps), jr.split(key_density, num_steps)),
    )
    return states


def _run_chains(chain_keys, make_step, initial_states, num_steps):
    if chain_keys.shape[0] == 1:
        single = _run_chain_scan(
            chain_keys[0],
            make_step,
            jax.tree_util.tree_map(lambda x: x[0], initial_states),
            num_steps,
        )
        return jax.tree_util.tree_map(lambda x: x[None, ...], single)

    return jax.pmap(
        _run_chain_scan,
        in_axes=(0, None, 0, None),
        static_broadcasted_argnums=(1, 3),
    )(chain_keys, make_step, initial_states, num_steps)


def _run_blackjax(
    mcmc_key: jnp.ndarray,
    make_algorithm: Callable,
    initial_positions,
    has_chain_axis: bool,
    num_chains: int,
    num_steps: int,
    transform_fn: Callable,
    num_warmup: int = 0,
) -> dict:
    mcmc_key, init_density_key = jr.split(mcmc_key)
    algorithm = make_algorithm(init_density_key)

    initial_states = (
        jax.vmap(algorithm.init)(initial_positions)
        if has_chain_axis
        else algorithm.init(initial_positions)
    )

    full_states = _run_chains(
        jr.split(mcmc_key, num_chains),
        lambda dk: make_algorithm(dk).step,
        initial_states,
        num_steps,
    )
    constrained = jax.jit(jax.vmap(jax.vmap(transform_fn)))(full_states.position)

    if num_warmup == 0:
        return constrained
    return jax.vmap(lambda s: {k: v[num_warmup:] for k, v in s.items()})(constrained)


def init_model(
    rng_key: jnp.ndarray,
    model: Callable,
    *,
    model_args: tuple,
    model_kwargs: dict,
    init_strategy=init_to_median,
):
    """Like numpyro's ``initialize_model`` but returns a key-aware potential function.

    NumPyro's ``initialize_model`` fixes the seed when building the potential
    function, causing Common Random Numbers (CRNs): stochastic model components
    (particle filters, EnKFs) see the same random seed at every MCMC step.

    This function instead returns a ``potential_fn_gen`` whose potential functions
    accept an explicit ``density_key``, so a fresh key can be passed at each step.

    Returns:
        ``(init_params, potential_fn_gen, postprocess_fn)`` where
        ``potential_fn_gen(*args)`` returns ``potential_fn(position, density_key)``.
    """
    init_params, _, postprocess_fn, *_ = initialize_model(
        rng_key=rng_key,
        model=model,
        model_args=model_args,
        model_kwargs=model_kwargs,
        dynamic_args=True,
        init_strategy=init_strategy,
    )

    def potential_fn_gen(*dynamic_args, **dynamic_kwargs):
        def potential_fn(position: dict, density_key: jnp.ndarray) -> jnp.ndarray:
            seeded_model = handlers.seed(model, density_key)
            return potential_energy(
                seeded_model, dynamic_args, dynamic_kwargs, position
            )

        return potential_fn

    return init_params, potential_fn_gen, postprocess_fn


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

    init_params, potential_fn_gen, postprocess_fn = init_model(
        rng_key=init_keys,
        model=model,
        model_args=(obs_times, obs_values, ctrl_times, ctrl_values, *model_args),
        model_kwargs=model_kwargs,
        init_strategy=mcmc_config.init_strategy,
    )
    initial_positions = init_params.z
    has_chain_axis = _has_chain_axis(initial_positions, mcmc_config.num_chains)

    potential_fn = potential_fn_gen(
        obs_times, obs_values, ctrl_times, ctrl_values, *model_args, **model_kwargs
    )
    transform_fn = postprocess_fn(
        obs_times, obs_values, ctrl_times, ctrl_values, *model_args, **model_kwargs
    )

    def make_logdensity(density_key):
        return lambda position: -potential_fn(position, density_key)

    if isinstance(mcmc_config, NUTSConfig):
        rng_key, warmup_key, warmup_density_key, mcmc_key = jr.split(rng_key, 4)
        warmup = blackjax.window_adaptation(
            blackjax.nuts, make_logdensity(warmup_density_key)
        )
        warmup_position = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )
        ((_, warmup_parameters), _) = warmup.run(
            warmup_key,
            warmup_position,
            num_steps=mcmc_config.num_warmup,  # ty: ignore[unknown-argument]
        )

        def make_nuts(density_key):
            return blackjax.nuts(make_logdensity(density_key), **warmup_parameters)

        return _run_blackjax(
            mcmc_key=mcmc_key,
            make_algorithm=make_nuts,
            initial_positions=initial_positions,
            has_chain_axis=has_chain_axis,
            num_chains=mcmc_config.num_chains,
            num_steps=mcmc_config.num_samples,
            transform_fn=transform_fn,
        )

    if isinstance(mcmc_config, HMCConfig):
        ref_position = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )

        if mcmc_config.adapt:
            rng_key, warmup_key, warmup_density_key, mcmc_key = jr.split(rng_key, 4)
            warmup = blackjax.window_adaptation(
                blackjax.hmc,
                make_logdensity(warmup_density_key),
                num_integration_steps=mcmc_config.num_steps,
            )
            ((_, warmup_parameters), _) = warmup.run(
                warmup_key,
                ref_position,
                num_steps=mcmc_config.num_warmup,  # ty: ignore[unknown-argument]
            )

            def make_hmc(density_key):
                return blackjax.hmc(make_logdensity(density_key), **warmup_parameters)

            return _run_blackjax(
                mcmc_key=mcmc_key,
                make_algorithm=make_hmc,
                initial_positions=initial_positions,
                has_chain_axis=has_chain_axis,
                num_chains=mcmc_config.num_chains,
                num_steps=mcmc_config.num_samples,
                transform_fn=transform_fn,
            )
        else:
            flat, _ = ravel_pytree(ref_position)
            inv_mass_matrix = jnp.eye(flat.shape[0])

            def make_hmc(density_key):
                return blackjax.hmc(
                    make_logdensity(density_key),
                    mcmc_config.step_size,
                    inv_mass_matrix,
                    mcmc_config.num_steps,
                )

            rng_key, mcmc_key = jr.split(rng_key)
            return _run_blackjax(
                mcmc_key=mcmc_key,
                make_algorithm=make_hmc,
                initial_positions=initial_positions,
                has_chain_axis=has_chain_axis,
                num_chains=mcmc_config.num_chains,
                num_steps=mcmc_config.num_samples + mcmc_config.num_warmup,
                transform_fn=transform_fn,
                num_warmup=mcmc_config.num_warmup,
            )

    if isinstance(mcmc_config, SGLDConfig):
        initial_positions = (
            initial_positions
            if has_chain_axis
            else jax.tree_util.tree_map(lambda x: x[None, ...], initial_positions)
        )

        def _run_sgld_chain(chain_key, init_position):
            total_steps = mcmc_config.num_warmup + mcmc_config.num_samples
            step_ids = jnp.arange(1, total_steps + 1, dtype=jnp.float32)
            step_sizes = mcmc_config.step_size * step_ids ** (
                -mcmc_config.schedule_power
            )

            key_step, key_density = jr.split(chain_key)
            step_keys = jr.split(key_step, total_steps)
            density_keys = jr.split(key_density, total_steps)

            def grad_estimator(position, density_key):
                return jax.grad(make_logdensity(density_key))(position)

            sgld = blackjax.sgld(grad_estimator)

            def _one_step(position, inputs):
                key_t, step_size_t, density_key_t = inputs
                next_position = sgld.step(key_t, position, density_key_t, step_size_t)
                return next_position, next_position

            _, chain_positions = jax.lax.scan(
                _one_step,
                sgld.init(init_position),
                (step_keys, step_sizes, density_keys),
            )
            post_warmup = jax.tree_util.tree_map(
                lambda x: x[mcmc_config.num_warmup :], chain_positions
            )
            return jax.vmap(transform_fn)(post_warmup)

        rng_key, mcmc_key = jr.split(rng_key)
        return jax.vmap(_run_sgld_chain)(
            jr.split(mcmc_key, mcmc_config.num_chains), initial_positions
        )

    if isinstance(mcmc_config, MALAConfig):

        def make_mala(density_key):
            return blackjax.mala(
                make_logdensity(density_key), step_size=mcmc_config.step_size
            )

        rng_key, mcmc_key = jr.split(rng_key)
        return _run_blackjax(
            mcmc_key=mcmc_key,
            make_algorithm=make_mala,
            initial_positions=initial_positions,
            has_chain_axis=has_chain_axis,
            num_chains=mcmc_config.num_chains,
            num_steps=mcmc_config.num_samples + mcmc_config.num_warmup,
            transform_fn=transform_fn,
            num_warmup=mcmc_config.num_warmup,
        )

    raise ValueError(f"Invalid MCMC config: {mcmc_config}")
