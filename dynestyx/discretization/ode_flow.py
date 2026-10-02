"""Numerical discrete transitions induced by deterministic ODE flows."""

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx.inference.configs.discretizer import ODEFlowConfig
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
)
from dynestyx.solvers import solve_ode_interval


class _ODEFlowStateEvolution(DiscreteTimeStateEvolution):
    """Discrete transition obtained by numerically integrating an ODE."""

    _dynestyx_discretizer_preserves_state_shape: ClassVar[bool] = True
    cte: DeterministicContinuousTimeStateEvolution
    config: ODEFlowConfig = eqx.field(static=True)

    def __init__(
        self,
        cte: DeterministicContinuousTimeStateEvolution,
        config: ODEFlowConfig,
    ):
        self.cte = cte
        self.config = config

    def __call__(self, x, u, t_now, t_next):
        endpoint = solve_ode_interval(
            self.cte,
            initial_state=x,
            t0=t_now,
            t1=t_next,
            u=u,
            diffeqsolve_settings=self.config.simulator_config.diffeqsolve_settings,
        )
        event_dim = 0 if jnp.ndim(endpoint) == 0 else 1
        if self.config.jitter_scale == 0.0:
            return dist.Delta(endpoint, event_dim=event_dim)
        scale = jnp.asarray(self.config.jitter_scale, dtype=endpoint.dtype)
        return dist.Normal(endpoint, scale).to_event(event_dim)
