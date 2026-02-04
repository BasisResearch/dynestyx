"""Discretization schemes for converting continuous-time state evolution to discrete-time."""

import jax.numpy as jnp
import numpyro.distributions as dist

from dsx.dynamical_models import (
    DiscreteTimeStateEvolution,
    ContinuousTimeStateEvolution,
)


class _EulerMaruyamaDiscreteEvolution(DiscreteTimeStateEvolution):
    """x_{t+1} ~ N(x + drift*dt, (L@Q@L.T)*dt)."""

    def __init__(self, cte: ContinuousTimeStateEvolution):
        self.cte = cte

    def __call__(self, x, u, t_now, t_next):
        delta_t = t_next - t_now
        drift = self.cte.drift(x, u, t_now)
        L_fn = getattr(self.cte, "diffusion_coefficient", None)
        if L_fn is None:
            raise AttributeError(
                "ContinuousTimeStateEvolution must define diffusion_coefficient."
            )
        L = L_fn(x, u, t_now) if callable(L_fn) else L_fn
        Q_fn = getattr(self.cte, "diffusion_covariance", jnp.eye(x.shape[-1]))
        Q = Q_fn(x, u, t_now) if callable(Q_fn) else Q_fn
        mean = x + drift * delta_t
        cov = (L @ Q @ L.T) * delta_t
        return dist.MultivariateNormal(loc=mean, covariance_matrix=cov)


def euler_maruyama(cte: ContinuousTimeStateEvolution) -> DiscreteTimeStateEvolution:
    """Discretize continuous-time state evolution via Euler-Maruyama. (CTSE) -> DTSE.

    Args:
        cte: ContinuousTimeStateEvolution to discretize.
    Returns:
        DiscreteTimeStateEvolution: The discretized state evolution.

    Note:
        No dt is passed; it is set to t_next - t_now in the __call__ method.

    How it works:
        x_{t+1} ~ N(x_t + drift * delta_t, (L@Q@L.T)*delta_t)
        where:
            x_t is the current state
            drift is the drift function
            L is the diffusion coefficient
            Q is the diffusion covariance
            delta_t is the time step between timepoints (t_next - t_now)
    """
    return _EulerMaruyamaDiscreteEvolution(cte)
