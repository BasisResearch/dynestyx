from numpyro.infer import HMC, MCMC, NUTS

from dynestyx.inference.integrations.blackjax import run_blackjax_mcmc
from dynestyx.inference.mcmc_configs import (
    AdjustedMCLMCDynamicConfig,
    HMCConfig,
    MALAConfig,
    NUTSConfig,
    SGLDConfig,
)


class MCMCInference:
    """Provides a high-level interface for MCMC inference, consistent between NumPyro and BlackJAX backends.

    Models must take in `obs_times`, `obs_values`, `ctrl_times`, `ctrl_values` as arguments (and optionally, `*model_args`, `**model_kwargs`).

    Attributes:
        mcmc_config: Sampler configuration dataclass (`NUTSConfig`,
            `HMCConfig`, `SGLDConfig`, `MALAConfig`, or
            `AdjustedMCLMCDynamicConfig`).
        model: Callable probabilistic model with signature
            `model(obs_times=..., obs_values=..., ctrl_times=..., ctrl_values=..., *model_args, **model_kwargs)`.
    """

    def __init__(self, mcmc_config, model):
        self.mcmc_config = mcmc_config
        self.model = model

    def run(
        self,
        rng_key,
        obs_times,
        obs_values,
        ctrl_times=None,
        ctrl_values=None,
        *model_args,
        **model_kwargs,
    ):
        """Run inference and return posterior samples.

        Args:
            rng_key: JAX PRNG key.
            obs_times: Observation times.
            obs_values: Observation values.
            ctrl_times: Control times.
            ctrl_values: Control values.
            *model_args: Additional positional arguments passed to `model`.
            **model_kwargs: Additional keyword arguments passed to `model`.

        Returns:
            Dict-like pytree of posterior samples.
        """

        if self.mcmc_config.mcmc_source == "numpyro":
            return _numpyro_mcmc(
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                model=self.model,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                *model_args,
                **model_kwargs,
            )
        elif self.mcmc_config.mcmc_source == "blackjax":
            return _blackjax_mcmc(
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                model=self.model,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                *model_args,
                **model_kwargs,
            )
        else:
            raise ValueError(f"Invalid MCMC source: {self.mcmc_config.mcmc_source}")


def _numpyro_mcmc(
    mcmc_config,
    rng_key,
    model,
    obs_times,
    obs_values,
    ctrl_times=None,
    ctrl_values=None,
    *model_args,
    **model_kwargs,
):
    """Run NumPyro-based MCMC (`NUTS` or `HMC`) and return samples."""
    if isinstance(mcmc_config, NUTSConfig):
        mcmc = MCMC(
            NUTS(model),
            num_warmup=mcmc_config.num_warmup,
            num_samples=mcmc_config.num_samples,
            num_chains=mcmc_config.num_chains,
        )
    elif isinstance(mcmc_config, HMCConfig):
        mcmc = MCMC(
            HMC(model),
            num_warmup=mcmc_config.num_warmup,
            num_samples=mcmc_config.num_samples,
            step_size=mcmc_config.step_size,
            num_steps=mcmc_config.num_steps,
            num_chains=mcmc_config.num_chains,
        )
    else:
        raise ValueError(f"Invalid MCMC config: {mcmc_config}")
    mcmc.run(
        rng_key,
        obs_times,
        obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        *model_args,
        **model_kwargs,
    )
    return mcmc.get_samples()


def _blackjax_mcmc(
    mcmc_config,
    rng_key,
    model,
    obs_times,
    obs_values,
    ctrl_times=None,
    ctrl_values=None,
    *model_args,
    **model_kwargs,
):
    """Run BlackJAX-based inference via the BlackJAX integration module."""
    if not isinstance(
        mcmc_config,
        NUTSConfig | HMCConfig | SGLDConfig | MALAConfig | AdjustedMCLMCDynamicConfig,
    ):
        raise ValueError(f"Invalid MCMC config: {mcmc_config}")
    return run_blackjax_mcmc(
        mcmc_config=mcmc_config,
        rng_key=rng_key,
        model=model,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        *model_args,
        **model_kwargs,
    )
