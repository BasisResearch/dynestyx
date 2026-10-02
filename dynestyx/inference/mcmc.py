from collections.abc import Callable

import jax
import jax.numpy as jnp
from numpyro.infer import HMC, MCMC, NUTS

from dynestyx.inference.configs.mcmc import (
    AdaptiveMetropolisConfig,
    BaseMCMCConfig,
    HMCConfig,
    MALAConfig,
    NUTSConfig,
    SGLDConfig,
)
from dynestyx.inference.integrations.blackjax import (
    run_blackjax_mcmc_with_diagnostics,
)


class MCMCInference:
    """Provides a high-level interface for MCMC inference, consistent between NumPyro and BlackJAX backends.

    Models must take in `obs_times`, `obs_values`, `ctrl_times`, `ctrl_values` as arguments (and optionally, `*model_args`, `**model_kwargs`).

    Attributes:
        mcmc_config: Sampler configuration dataclass (`NUTSConfig`,
            `HMCConfig`, `AdaptiveMetropolisConfig`, `SGLDConfig`, or
            `MALAConfig`).
        model: Callable probabilistic model with signature
            `model(obs_times=..., obs_values=..., ctrl_times=..., ctrl_values=..., *model_args, **model_kwargs)`.
    """

    def __init__(self, mcmc_config: BaseMCMCConfig, model: Callable):
        self.mcmc_config = mcmc_config
        self.model = model
        self._diagnostics: dict[str, jax.Array] | None = None

    def get_diagnostics(self) -> dict[str, jax.Array]:
        """Return compact diagnostics from the most recent successful run.

        NUTS reports ``mean_acceptance_rate`` and ``num_divergences`` per
        chain. Adaptive Metropolis reports ``mean_acceptance_rate`` and
        ``final_proposal_scale`` per chain and unconstrained coordinate.

        Raises:
            RuntimeError: If inference has not completed successfully.
        """
        if self._diagnostics is None:
            raise RuntimeError("No MCMC diagnostics are available; call run() first")
        return self._diagnostics.copy()

    def run(
        self,
        rng_key: jnp.ndarray,
        obs_times: jnp.ndarray,
        obs_values: jnp.ndarray,
        ctrl_times: jnp.ndarray | None = None,
        ctrl_values: jnp.ndarray | None = None,
        *model_args,
        **model_kwargs,
    ) -> dict:
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

        self._diagnostics = None
        if self.mcmc_config.mcmc_source == "numpyro":
            samples, diagnostics = _numpyro_mcmc(  # type: ignore
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                model=self.model,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                *model_args,  # type: ignore
                **model_kwargs,
            )
        elif self.mcmc_config.mcmc_source == "blackjax":
            samples, diagnostics = _blackjax_mcmc(  # type: ignore
                mcmc_config=self.mcmc_config,
                rng_key=rng_key,
                model=self.model,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                *model_args,  # type: ignore
                **model_kwargs,
            )
        else:
            raise ValueError(f"Invalid MCMC source: {self.mcmc_config.mcmc_source}")
        self._diagnostics = diagnostics
        return samples


def _numpyro_mcmc(
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
    """Run NumPyro MCMC and return samples plus compact diagnostics."""
    if isinstance(mcmc_config, NUTSConfig):
        mcmc = MCMC(
            NUTS(
                model,
                init_strategy=mcmc_config.init_strategy,
                target_accept_prob=mcmc_config.target_acceptance_rate,
            ),
            num_warmup=mcmc_config.num_warmup,
            num_samples=mcmc_config.num_samples,
            num_chains=mcmc_config.num_chains,
        )
    elif isinstance(mcmc_config, HMCConfig):
        mcmc = MCMC(
            HMC(
                model,
                step_size=mcmc_config.step_size,
                num_steps=mcmc_config.num_steps,
                adapt_step_size=mcmc_config.adapt,
                adapt_mass_matrix=mcmc_config.adapt,
                init_strategy=mcmc_config.init_strategy,
            ),
            num_warmup=mcmc_config.num_warmup,
            num_samples=mcmc_config.num_samples,
            num_chains=mcmc_config.num_chains,
        )
    else:
        raise ValueError(f"Invalid MCMC config: {mcmc_config}")
    run_kwargs = dict(model_kwargs)
    if isinstance(mcmc_config, NUTSConfig):
        run_kwargs["extra_fields"] = ("accept_prob", "diverging")
    mcmc.run(  # type: ignore
        rng_key,
        obs_times,
        obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        *model_args,
        **run_kwargs,
    )
    diagnostics = {}
    if isinstance(mcmc_config, NUTSConfig):
        extra_fields = mcmc.get_extra_fields(group_by_chain=True)
        diagnostics = {
            "mean_acceptance_rate": jnp.mean(extra_fields["accept_prob"], axis=1),
            "num_divergences": jnp.sum(extra_fields["diverging"], axis=1),
        }
    return mcmc.get_samples(), diagnostics


def _blackjax_mcmc(
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
    """Run BlackJAX inference via the BlackJAX integration module."""
    if not isinstance(
        mcmc_config,
        NUTSConfig | HMCConfig | AdaptiveMetropolisConfig | SGLDConfig | MALAConfig,
    ):
        raise ValueError(f"Invalid MCMC config: {mcmc_config}")
    return run_blackjax_mcmc_with_diagnostics(  # type: ignore
        mcmc_config=mcmc_config,
        rng_key=rng_key,
        model=model,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        *model_args,  # type: ignore
        **model_kwargs,
    )
