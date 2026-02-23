import jax.numpy as jnp
import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax import vmap

from dynestyx.handlers import HandlesSelf, sample
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    DynamicalModel,
)
from dynestyx.types import FunctionOfTime


class _EulerMaruyamaDiscreteEvolution(DiscreteTimeStateEvolution):
    """x_{t+1} ~ N(x + drift*dt, (L@Q@L.T)*dt)."""

    def __init__(self, cte: ContinuousTimeStateEvolution):
        self.cte = cte

    def __call__(self, x, u, t_now, t_next):
        """
        Discretize continuous-time state evolution via Euler-Maruyama. (CTSE) -> DTSE.

        We step from t_now to t_next for each timepoint provided (optionally just 1 timepoint provided).
        The main use case of providing multiple timepoints is when paired with DiracDeltaObservation that
        allows temporal independence between observations, which allows us to step through all timepoints at once (creating big speedups).

            Args:
                x: (dim_state,) or (dim_state, num_timepoints)
                u: (dim_control,) or (dim_control, num_timepoints)
                t_now: (1,) or (num_timepoints,)
                t_next: (1,) or (num_timepoints,)

            Returns:
                dist: MultivariateNormal distribution
                    - loc: (dim_state, num_timepoints) or (dim_state)
                    - covariance_matrix: (dim_state, dim_state, num_timepoints) or (dim_state, dim_state)
        """

        squeezed = False
        if x.ndim == 1:
            squeezed = True
            x = x[:, None]  # (dim_state, 1) state
        if u is not None:
            if u.ndim == 1:
                u = u[:, None]  # (dim_control, 1) control
        if t_now.ndim == 0:
            t_now = t_now[None]  # (1,) timepoint
        if t_next.ndim == 0:
            t_next = t_next[None]  # (1,) timepoint

        def _step(_x, _u, _t_now, _t_next):
            _dt = _t_next - _t_now
            drift = self.cte.total_drift(_x, _u, _t_now)
            x_pred_mean = _x + drift * _dt
            L = self.cte.diffusion_coefficient(_x, _u, _t_now)
            Q = jnp.eye(self.cte.bm_dim)
            x_pred_cov = L @ Q @ L.T * _dt
            return x_pred_mean, x_pred_cov

        if u is None:
            loc, cov = vmap(_step, in_axes=(1, None, 0, 0))(x, None, t_now, t_next)
        else:
            loc, cov = vmap(_step, in_axes=(1, 1, 0, 0))(x, u, t_now, t_next)

        # If we lifted from unbatched, return unbatched dist shapes
        if squeezed:
            loc = loc[0]
            cov = cov[0]

        return dist.MultivariateNormal(loc=loc, covariance_matrix=cov)


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


class Discretizer(ObjectInterpretation, HandlesSelf):
    """
    Discretize a continuous-time state evolution to a discrete-time state evolution.
    Args:
        discretize: Callable (CTSE) -> DTSE. Defaults to euler_maruyama.
    """

    def __init__(self, discretize=euler_maruyama):
        super().__init__()
        self.discretize = discretize

    @implements(sample)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> FunctionOfTime:
        if isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            discrete_evolution = self.discretize(dynamics.state_evolution)
            dynamics = DynamicalModel(
                initial_condition=dynamics.initial_condition,
                state_evolution=discrete_evolution,
                observation_model=dynamics.observation_model,
                control_model=dynamics.control_model,
                state_dim=dynamics.state_dim,
                observation_dim=dynamics.observation_dim,
                control_dim=dynamics.control_dim,
            )
        return fwd(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
