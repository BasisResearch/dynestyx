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

from dynestyx.inference.configs.mcmc import (
    AdaptiveMetropolisConfig,
    BaseMCMCConfig,
    HMCConfig,
    MALAConfig,
    NUTSConfig,
    SGLDConfig,
)
from dynestyx.inference.integrations.blackjax.adaptive_metropolis import (
    adaptive_metropolis,
    resolve_proposal_scale,
)


def _has_chain_axis(initial_positions, num_chains: int) -> bool:
    leaves = jax.tree_util.tree_leaves(initial_positions)
    return (
        len(leaves) > 0
        and hasattr(leaves[0], "shape")
        and len(leaves[0].shape) >= 1
        and leaves[0].shape[0] == num_chains
    )


def _run_chain_scan(
    rng_key,
    make_step,
    initial_state,
    num_steps,
    info_fn=None,
):
    """Scan ``num_steps`` MCMC steps, passing a fresh density key to each."""
    if info_fn is None:
        info_fn = lambda state, info: ()

    def one_step(state, keys):
        mcmc_key, density_key = keys
        state, info = make_step(density_key)(mcmc_key, state)
        return state, (state.position, info_fn(state, info))

    key_mcmc, key_density = jr.split(rng_key)
    final_state, result = jax.lax.scan(
        one_step,
        initial_state,
        (jr.split(key_mcmc, num_steps), jr.split(key_density, num_steps)),
    )
    return final_state, result


def _run_chains(chain_keys, run_chain, initial_states):
    """Apply one BlackJAX chain runner with standard chain semantics."""
    if chain_keys.shape[0] == 1:
        state = jax.tree_util.tree_map(lambda x: x[0], initial_states)
        result = run_chain(chain_keys[0], state)
        return jax.tree_util.tree_map(lambda x: x[None, ...], result)
    if jax.local_device_count() >= chain_keys.shape[0]:
        return jax.pmap(run_chain)(chain_keys, initial_states)
    return jax.vmap(run_chain)(chain_keys, initial_states)


def _run_blackjax(
    mcmc_key: jnp.ndarray,
    make_algorithm: Callable,
    initial_positions,
    has_chain_axis: bool,
    num_chains: int,
    num_steps: int,
    transform_fn: Callable,
    num_warmup: int = 0,
    info_fn: Callable | None = None,
    diagnostics_fn: Callable | None = None,
) -> tuple[dict, dict[str, jax.Array]]:
    mcmc_key, init_density_key = jr.split(mcmc_key)
    algorithm = make_algorithm(init_density_key)

    initial_states = (
        jax.vmap(algorithm.init)(initial_positions)  # type: ignore[call-arg]
        if has_chain_axis
        else algorithm.init(initial_positions)  # type: ignore[call-arg]
    )

    chain_keys = jr.split(mcmc_key, num_chains)
    make_step = lambda dk: make_algorithm(dk).step

    def run_chain(key, state):
        return _run_chain_scan(
            key,
            make_step,
            state,
            num_steps,
            info_fn,
        )

    final_states, (positions, info) = _run_chains(
        chain_keys,
        run_chain,
        initial_states,
    )
    diagnostics = diagnostics_fn(final_states, info) if diagnostics_fn else {}
    constrained = jax.jit(jax.vmap(jax.vmap(transform_fn)))(positions)

    if num_warmup == 0:
        return constrained, diagnostics
    samples = jax.vmap(lambda s: {k: v[num_warmup:] for k, v in s.items()})(constrained)
    return samples, diagnostics


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


def run_blackjax_mcmc_with_diagnostics(
    mcmc_config: BaseMCMCConfig,
    rng_key: jnp.ndarray,
    model: Callable,
    obs_times: jnp.ndarray,
    obs_values: jnp.ndarray,
    ctrl_times: jnp.ndarray | None = None,
    ctrl_values: jnp.ndarray | None = None,
    *model_args,
    **model_kwargs,
) -> tuple[dict, dict[str, jax.Array]]:
    """Run BlackJAX inference and return samples plus compact diagnostics."""
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
            blackjax.nuts,
            make_logdensity(warmup_density_key),
            target_acceptance_rate=mcmc_config.target_acceptance_rate,
        )
        warmup_position = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )
        ((_, warmup_parameters), _) = warmup.run(  # type: ignore
            warmup_key, warmup_position, num_steps=mcmc_config.num_warmup
        )

        def make_nuts(density_key):
            return blackjax.nuts(make_logdensity(density_key), **warmup_parameters)

        def nuts_info(state, info):
            del state
            return info.acceptance_rate, info.is_divergent

        def nuts_diagnostics(final_state, info):
            del final_state
            acceptance_rate, is_divergent = info
            return {
                "mean_acceptance_rate": jnp.mean(acceptance_rate, axis=1),
                "num_divergences": jnp.sum(is_divergent, axis=1),
            }

        return _run_blackjax(
            mcmc_key=mcmc_key,
            make_algorithm=make_nuts,
            initial_positions=initial_positions,
            has_chain_axis=has_chain_axis,
            num_chains=mcmc_config.num_chains,
            num_steps=mcmc_config.num_samples,
            transform_fn=transform_fn,
            info_fn=nuts_info,
            diagnostics_fn=nuts_diagnostics,
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
            ((_, warmup_parameters), _) = warmup.run(  # type: ignore
                warmup_key, ref_position, num_steps=mcmc_config.num_warmup
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
        samples = jax.vmap(_run_sgld_chain)(
            jr.split(mcmc_key, mcmc_config.num_chains), initial_positions
        )
        return samples, {}

    if isinstance(mcmc_config, AdaptiveMetropolisConfig):
        reference_position = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )
        reference_flat, unravel_fn = ravel_pytree(reference_position)
        flat_initial_positions = (
            jax.vmap(lambda x: ravel_pytree(x)[0])(initial_positions)
            if has_chain_axis
            else reference_flat[None, ...]
        )
        proposal_scale = resolve_proposal_scale(
            jnp.asarray(mcmc_config.initial_proposal_scale),
            reference_flat.size,
        ).astype(reference_flat.dtype)

        def make_adaptive_metropolis(density_key):
            logdensity_fn = make_logdensity(density_key)

            def flat_logdensity(position):
                return logdensity_fn(unravel_fn(position))

            return adaptive_metropolis(
                flat_logdensity,
                proposal_scale,
                target_acceptance_rate=mcmc_config.target_acceptance_rate,
                adaptation_rate=mcmc_config.adaptation_rate,
                max_adaptation=mcmc_config.max_adaptation,
                num_warmup=mcmc_config.num_warmup,
            )

        def flat_transform(position):
            return transform_fn(unravel_fn(position))

        def acceptance_info(state, info):
            del state
            return info.is_accepted

        def adaptive_diagnostics(final_state, is_accepted):
            return {
                "mean_acceptance_rate": jnp.mean(
                    is_accepted[:, mcmc_config.num_warmup :], axis=1
                ),
                "final_proposal_scale": final_state.proposal_scale,
            }

        rng_key, mcmc_key = jr.split(rng_key)
        return _run_blackjax(
            mcmc_key=mcmc_key,
            make_algorithm=make_adaptive_metropolis,
            initial_positions=flat_initial_positions,
            has_chain_axis=True,
            num_chains=mcmc_config.num_chains,
            num_steps=mcmc_config.num_warmup + mcmc_config.num_samples,
            transform_fn=flat_transform,
            num_warmup=mcmc_config.num_warmup,
            info_fn=acceptance_info,
            diagnostics_fn=adaptive_diagnostics,
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
    samples, _ = run_blackjax_mcmc_with_diagnostics(
        mcmc_config,
        rng_key,
        model,
        obs_times,
        obs_values,
        ctrl_times,
        ctrl_values,
        *model_args,
        **model_kwargs,
    )
    return samples
