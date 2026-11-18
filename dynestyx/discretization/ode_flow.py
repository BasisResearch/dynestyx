"""Numerical discrete transitions induced by deterministic ODE flows."""

from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx.inference.configs.discretizer import ODEFlowConfig
from dynestyx.inference.configs.simulator import ODESimulatorConfig
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DeterministicStateEvolution,
    GaussianStateEvolution,
    ScalarCovariance,
)
from dynestyx.solvers import solve_ode_interval


class _ODEFlowEndpoint(eqx.Module):
    cte: DeterministicContinuousTimeStateEvolution
    solver_config: ODESimulatorConfig = eqx.field(static=True)

    def __call__(self, x, u, t_now, t_next):
        return solve_ode_interval(
            self.cte,
            initial_state=x,
            t0=t_now,
            t1=t_next,
            u=u,
            diffeqsolve_settings=self.solver_config.diffeqsolve_settings,
        )


class _ODEFlowStateEvolution:
    """Shared identity for transitions whose mean is a numerical ODE flow."""

    _dynestyx_discretizer_preserves_state_shape: ClassVar[bool] = True


class _DeterministicODEFlowStateEvolution(
    DeterministicStateEvolution, _ODEFlowStateEvolution
):
    pass


class _GaussianODEFlowStateEvolution(GaussianStateEvolution, _ODEFlowStateEvolution):
    def __call__(self, x, u, t_now, t_next):
        # Preserve independent Normal sampling for isotropic endpoint noise.
        # GaussianRelaxation can replace the covariance with a full matrix.
        if isinstance(self.cov, ScalarCovariance):
            loc = self.F(x, u, t_now, t_next)
            sd = jnp.sqrt(self.cov.variance).astype(loc.dtype)
            if loc.ndim == 0:
                return dist.Normal(loc, sd)
            return dist.Normal(loc, sd[..., None]).to_event(1)
        return super().__call__(x, u, t_now, t_next)


def ode_flow_state_evolution(
    cte: DeterministicContinuousTimeStateEvolution, config: ODEFlowConfig
) -> DeterministicStateEvolution | GaussianStateEvolution:
    endpoint = _ODEFlowEndpoint(cte, config.simulator_config)
    if config.jitter_scale == 0:
        return _DeterministicODEFlowStateEvolution(F=endpoint)
    return _GaussianODEFlowStateEvolution(
        F=endpoint, cov=ScalarCovariance(sd=config.jitter_scale)
    )
