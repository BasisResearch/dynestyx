"""Discretization schemes for converting continuous-time state evolution to discrete-time."""

import jax.numpy as jnp
import numpyro.distributions as dist

from dsx.dynamical_models import (
    DiscreteTimeStateEvolution,
    ContinuousTimeStateEvolution,
)


class EulerMaruyamaDiscretization(DiscreteTimeStateEvolution):
    def __init__(self, continuous_time_evolution: ContinuousTimeStateEvolution):
        super().__init__()
        self.cte = continuous_time_evolution

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
