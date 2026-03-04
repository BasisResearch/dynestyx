from numpyro.infer import HMC, MCMC, NUTS

from dynestyx.inference.filters import Filter
from dynestyx.inference.integrations.blackjax import run_blackjax_mcmc
from dynestyx.inference.mcmc_configs import HMCConfig, NUTSConfig, SGLDConfig


class FilterBasedMCMC:
    """Run parameter inference with a filtering-based likelihood.

    This class wraps a model in `Filter(...)` and dispatches to the selected
    MCMC backend defined by `mcmc_config`.

    Attributes:
        filter_config: Configuration passed to `Filter`.
        mcmc_config: Sampler configuration dataclass (`NUTSConfig`,
            `HMCConfig`, or `SGLDConfig`).
        model: Callable probabilistic model with signature
            `model(obs_times=..., obs_values=..., *model_args)`.
    """

    def __init__(self, filter_config, mcmc_config, model):
        self.filter_config = filter_config
        self.mcmc_config = mcmc_config
        self.model = model

    def run(self, rng_key, obs_times, obs_values, *model_args):
        """Run inference and return posterior samples.

        Args:
            rng_key: JAX PRNG key.
            obs_times: Observation times.
            obs_values: Observation values.
            *model_args: Additional positional arguments passed to `model`.

        Returns:
            Dict-like pytree of posterior samples.
        """

        def data_conditioned_model(obs_times=None, obs_values=None):
            with Filter(self.filter_config):
                return self.model(
                    obs_times=obs_times, obs_values=obs_values, *model_args
                )

        if self.mcmc_config.mcmc_source == "numpyro":
            return _numpyro_mcmc(
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                data_conditioned_model=data_conditioned_model,
                obs_times=obs_times,
                obs_values=obs_values,
            )
        elif self.mcmc_config.mcmc_source == "blackjax":
            return _blackjax_mcmc(
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                data_conditioned_model=data_conditioned_model,
                obs_times=obs_times,
                obs_values=obs_values,
            )
        else:
            raise ValueError(f"Invalid MCMC source: {self.mcmc_config.mcmc_source}")


def _numpyro_mcmc(mcmc_config, rng_key, data_conditioned_model, obs_times, obs_values):
    """Run NumPyro-based MCMC (`NUTS` or `HMC`) and return samples."""
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


def _blackjax_mcmc(mcmc_config, rng_key, data_conditioned_model, obs_times, obs_values):
    """Run BlackJAX-based inference via the BlackJAX integration module."""
    if not isinstance(mcmc_config, NUTSConfig | HMCConfig | SGLDConfig):
        raise ValueError(f"Invalid MCMC config: {mcmc_config}")
    return run_blackjax_mcmc(
        mcmc_config=mcmc_config,
        rng_key=rng_key,
        data_conditioned_model=data_conditioned_model,
        obs_times=obs_times,
        obs_values=obs_values,
    )
