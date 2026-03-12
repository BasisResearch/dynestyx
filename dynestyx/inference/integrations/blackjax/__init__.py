"""BlackJAX backend for MCMC and SGMCMC inference."""

from dynestyx.inference.integrations.blackjax.mcmc import run_blackjax_mcmc

__all__ = ["run_blackjax_mcmc"]
