import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax import vmap

from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiscreteTimeStateEvolution,
    DynamicalModel,
)
from dynestyx.models.checkers import _infer_bm_dim
from dynestyx.types import FunctionOfTime


def _ensure_ctse_bm_dim(dynamics: DynamicalModel) -> DynamicalModel:
    """Infer and set bm_dim when CT dynamics are built under active plates."""
    if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
        return dynamics

    cte = dynamics.state_evolution
    if cte.diffusion_coefficient is None or cte.bm_dim is not None:
        return dynamics

    x0 = jnp.zeros((dynamics.state_dim,))
    u0 = None if dynamics.control_dim == 0 else jnp.zeros((dynamics.control_dim,))
    t0 = jnp.array(0.0) if dynamics.t0 is None else jnp.asarray(dynamics.t0)
    inferred_bm_dim = _infer_bm_dim(cte, dynamics.state_dim, x0, u0, t0)
    if inferred_bm_dim is not None:
        object.__setattr__(cte, "bm_dim", inferred_bm_dim)
    return dynamics


class _EulerMaruyamaDiscreteEvolution(DiscreteTimeStateEvolution):
    """x_{t+1} ~ N(x + drift*dt, (L@Q@L.T)*dt)."""

    cte: ContinuousTimeStateEvolution

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
            if self.cte.bm_dim is None:
                raise ValueError(
                    "ContinuousTimeStateEvolution.bm_dim is not set. "
                    "Construct dynamics via DynamicalModel before discretization."
                )
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
    """Discretize continuous-time state evolution via Euler-Maruyama.

    Euler-Maruyama is a first-order discrete approximation of a continuous-time state evolution.
    It is popular, as it is simple and effective for simple models.
    The resulting discrete-time state evolution is linear and Gaussian.

    Args:
        cte: `ContinuousTimeStateEvolution` to discretize.
    Returns:
        DiscreteTimeStateEvolution: The discretized state evolution.

    Note:
        No dt is passed; it is set to t_next - t_now in the __call__ method.

    ??? note "Algorithm Reference"
        The Euler Maruyama is a first order discretization.
        The resulting discret-time state evolution is approximated as

        x_{t+1} ~ N(x_t + drift * delta_t, (L@Q@L.T)*delta_t)

        where:
            x_t is the current state
            drift is the drift function
            L is the diffusion coefficient
            Q is the diffusion covariance
            delta_t is the time step between timepoints (t_next - t_now)

        This is the first-order Ito-Taylor approximation.

        References:
            - This is the first-order Ito-Taylor approximation, discussed in Chapter 9.2 of: Särkkä, S., & Solin, A. (2019).
                Applied Stochastic Differential Equations. Cambridge University Press.
                [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).
    """
    return _EulerMaruyamaDiscreteEvolution(cte)


class Discretizer(ObjectInterpretation, HandlesSelf):
    """
    Performs discretization of a continuous-time state evolution, converting it to a discrete-time state evolution.

    A `Discretizer` object should be used as a context manager around a call to a model with a `dsx.sample(...)`
    statement to discretize a continuous-time state evolution to a discrete-time state evolution. The `Discretizer`
    should be at a lower (i.e. inner) level in the current context stack than any inference (e.g., `Filter` or `Simulator`)
    objects.

    ??? example "Using a Euler Maruyama Discretizer"
        ```python
        import dynestyx as dsx
        from dynestyx.discretizers import Discretizer, euler_maruyama
        from dynestyx.inference.filters import Filter, EKFConfig
        from dynestyx.models import (
            ContinuousTimeStateEvolution,
            DiscreteTimeStateEvolution,
            DynamicalModel,
        )

        def model_with_ctse(obs_times=None, obs_values=None):
            dynamics = DynamicalModel(
                control_dim=0,
                initial_condition=dist.MultivariateNormal(
                    loc=jnp.zeros(state_dim),
                    covariance_matrix=jnp.eye(state_dim),
                ),
                state_evolution=ContinuousTimeStateEvolution(
                    drift=lambda x, u, t: x,
                    diffusion_coefficient=lambda x, u, t: jnp.eye(state_dim, bm_dim),
                ),
                observation_model=lambda x, u, t: dist.MultivariateNormal(
                    x,
                    0.1**2 * jnp.eye(observation_dim),
                ),
            )
            return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

        def discretized_data_conditioned_model():
            # We use a discrete-time filter now
            with Filter(filter_config=EKFConfig()):
                with Discretizer(discretize=euler_maruyama):
                    return model_with_ctse(obs_times=obs_times, obs_values=obs_values)
        ```

    ??? note "Algorithm Reference"
        For an overview of discretization methods for SDEs, see Chapter 9 of: Särkkä, S., & Solin, A. (2019).
        Applied Stochastic Differential Equations. Cambridge University Press.
        [Available Online](https://users.aalto.fi/~asolin/sde-book/sde-book.pdf).

    Attributes:
        discretize: A callable that converts a continuous-time state evolution to a discrete-time state evolution. Defaults to euler_maruyama.
    """

    def __init__(self, discretize=euler_maruyama):
        super().__init__()
        self.discretize = discretize

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> FunctionOfTime:
        if isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            dynamics = _ensure_ctse_bm_dim(dynamics)
            discrete_evolution = self.discretize(dynamics.state_evolution)
            dynamics = DynamicalModel(
                initial_condition=dynamics.initial_condition,
                state_evolution=discrete_evolution,
                observation_model=dynamics.observation_model,
                control_model=dynamics.control_model,
                control_dim=dynamics.control_dim,
                t0=dynamics.t0,
            )
        return fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
