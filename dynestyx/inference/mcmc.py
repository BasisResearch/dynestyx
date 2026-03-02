import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from numpyro.infer import HMC, MCMC, NUTS, init_to_median
from numpyro.infer.util import initialize_model

from dynestyx.inference.filters import Filter
from dynestyx.inference.mcmc_configs import HMCConfig, NUTSConfig


class FilterBasedMCMC:
    def __init__(self, filter_config, mcmc_config, model):
        self.filter_config = filter_config
        self.mcmc_config = mcmc_config
        self.model = model

    def run(self, rng_key, obs_times, obs_values, *model_args):
        def data_conditioned_model(obs_times=None, obs_values=None):
            with Filter(self.filter_config):
                return self.model(
                    obs_times=obs_times, obs_values=obs_values, *model_args
                )

        if self.mcmc_config.mcmc_source == "numpyro":
            return numpyro_mcmc(
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                data_conditioned_model=data_conditioned_model,
                obs_times=obs_times,
                obs_values=obs_values,
            )
        elif self.mcmc_config.mcmc_source == "blackjax":
            return blackjax_mcmc(
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                data_conditioned_model=data_conditioned_model,
                obs_times=obs_times,
                obs_values=obs_values,
            )
        else:
            raise ValueError(f"Invalid MCMC source: {self.mcmc_config.mcmc_source}")


def numpyro_mcmc(mcmc_config, rng_key, data_conditioned_model, obs_times, obs_values):
    if isinstance(mcmc_config, NUTSConfig):
        mcmc = MCMC(
            NUTS(data_conditioned_model),
            num_warmup=mcmc_config.num_warmup,
            num_samples=mcmc_config.num_samples,
            num_chains=mcmc_config.num_chains,
        )
    elif isinstance(mcmc_config, HMCConfig):
        mcmc = MCMC(
            HMC(data_conditioned_model),
            num_warmup=mcmc_config.num_warmup,
            num_samples=mcmc_config.num_samples,
            step_size=mcmc_config.step_size,
            num_steps=mcmc_config.num_steps,
            num_chains=mcmc_config.num_chains,
        )
    else:
        raise ValueError(f"Invalid MCMC config: {mcmc_config}")
    mcmc.run(rng_key, obs_times, obs_values)
    return mcmc.get_samples()


def blackjax_mcmc(mcmc_config, rng_key, data_conditioned_model, obs_times, obs_values):
    rng_key, init_key_master = jr.split(rng_key)

    init_keys = jr.split(init_key_master, mcmc_config.num_chains)
    init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        rng_key=init_keys,
        model=data_conditioned_model,
        model_args=(obs_times, obs_values),
        dynamic_args=True,
        init_strategy=init_to_median,
    )

    logdensity_fn = lambda position: -potential_fn_gen(obs_times, obs_values)(position)

    if isinstance(mcmc_config, NUTSConfig):
        initial_positions = init_params.z

        warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_fn)

        leaves = jax.tree_util.tree_leaves(initial_positions)
        has_chain_axis = (
            len(leaves) > 0
            and hasattr(leaves[0], "shape")
            and len(leaves[0].shape) >= 1
            and leaves[0].shape[0] == mcmc_config.num_chains
        )
        initial_position_for_warmup = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )

        rng_key, warmup_key, mcmc_key = jr.split(rng_key, 3)
        ((_, warmup_parameters), _) = warmup.run(
            warmup_key,
            initial_position_for_warmup,
            num_steps=mcmc_config.num_warmup,
        )

        nuts = blackjax.nuts(logdensity_fn, **warmup_parameters)

        initial_states = (
            jax.vmap(nuts.init, in_axes=(0,))(initial_positions)
            if has_chain_axis
            else nuts.init(initial_positions)
        )

        def inference_loop(rng_key, kernel, initial_state, num_samples):

            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        inference_loop_multiple_chains = jax.pmap(
            inference_loop,
            in_axes=(0, None, 0, None),
            static_broadcasted_argnums=(1, 3),
        )

        mcmc_keys = jr.split(mcmc_key, mcmc_config.num_chains)

        full_states = inference_loop_multiple_chains(
            mcmc_keys, nuts.step, initial_states, mcmc_config.num_samples
        )

        def postprocess_fn_multiple_chains(states):
            constrained_positions = jax.jit(
                jax.vmap(jax.vmap(postprocess_fn(obs_times, obs_values)))
            )(states.position)
            return constrained_positions

        return postprocess_fn_multiple_chains(full_states)

    elif isinstance(mcmc_config, HMCConfig):
        initial_positions = init_params.z

        leaves = jax.tree_util.tree_leaves(initial_positions)
        has_chain_axis = (
            len(leaves) > 0
            and hasattr(leaves[0], "shape")
            and len(leaves[0].shape) >= 1
            and leaves[0].shape[0] == mcmc_config.num_chains
        )
        position_for_metric = (
            jax.tree_util.tree_map(lambda x: x[0], initial_positions)
            if has_chain_axis
            else initial_positions
        )
        flat_position, _ = ravel_pytree(position_for_metric)
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

        def inference_loop(rng_key, kernel, initial_state, num_samples):

            @jax.jit
            def one_step(state, rng_key):
                state, _ = kernel(rng_key, state)
                return state, state

            keys = jax.random.split(rng_key, num_samples)
            _, states = jax.lax.scan(one_step, initial_state, keys)

            return states

        inference_loop_multiple_chains = jax.pmap(
            inference_loop,
            in_axes=(0, None, 0, None),
            static_broadcasted_argnums=(1, 3),
        )
        rng_key, mcmc_key = jr.split(rng_key)
        mcmc_keys = jr.split(mcmc_key, mcmc_config.num_chains)

        full_states = inference_loop_multiple_chains(
            mcmc_keys,
            hmc.step,
            initial_states,
            mcmc_config.num_samples + mcmc_config.num_warmup,
        )

        def postprocess_fn_multiple_chains(states):
            constrained_positions = jax.jit(
                jax.vmap(jax.vmap(postprocess_fn(obs_times, obs_values)))
            )(states.position)

            def _remove_warmup(positions):
                return {k: v[mcmc_config.num_warmup :] for k, v in positions.items()}

            return jax.vmap(_remove_warmup)(constrained_positions)

        return postprocess_fn_multiple_chains(full_states)
