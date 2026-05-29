import numpyro.distributions as dist
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Real

from dynestyx.handlers import HandlesSelf, _infer_intp
from dynestyx.models import (
    DynamicalModel,
    GaussianStateEvolution,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.solvers import euler_maruyama_loc_cov
from dynestyx.types import FunctionOfTime


class EulerMaruyamaGaussianStateEvolution(GaussianStateEvolution):
    """`GaussianStateEvolution` backed by Euler-Maruyama moments."""

    cte: StochasticContinuousTimeStateEvolution

    def __init__(
        self,
        cte: StochasticContinuousTimeStateEvolution,
        F=None,
        cov=None,
    ):
        # Accept these for reconstruction paths, but derive both from `cte`.
        self.cte = cte

        def _loc(x, u, t_now, t_next):
            return euler_maruyama_loc_cov(cte, x, u, t_now, t_next)["loc"]

        def _cov(x, u, t_now, t_next):
            return euler_maruyama_loc_cov(cte, x, u, t_now, t_next)["cov"]

        super().__init__(
            F=_loc,
            cov=_cov,
        )

    def __call__(self, x, u, t_now, t_next):
        """Single-pass transition step (or batched time steps)."""
        em_result = euler_maruyama_loc_cov(self.cte, x, u, t_now, t_next)
        return dist.MultivariateNormal(
            loc=em_result["loc"], covariance_matrix=em_result["cov"]
        )


def euler_maruyama(
    cte: StochasticContinuousTimeStateEvolution,
) -> GaussianStateEvolution:
    """Discretize continuous-time state evolution via Euler-Maruyama.

    Euler-Maruyama is a first-order discrete approximation of a continuous-time
    SDE. The result is a `GaussianStateEvolution` with mean
    `x + drift(x,u,t)*dt` and covariance `(L@Q@L.T)*dt` (`Q = I`),
    where `dt = t_next - t_now`. The process covariance is **time-varying**
    (depends on `t_next - t_now`) and passed as a callable `cov`.

    Args:
        cte: `StochasticContinuousTimeStateEvolution` to discretize.
    Returns:
        GaussianStateEvolution: Discrete-time Gaussian transition with the
        same Euler–Maruyama semantics as before this refactor.

    Note:
        Each transition uses one Euler-Maruyama step with
        `dt = t_next - t_now`.

    ??? note "Algorithm Reference"
        The Euler Maruyama is a first order discretization.
        The resulting discrete-time state evolution is approximated as

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

    return EulerMaruyamaGaussianStateEvolution(cte)


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
            FullDiffusion,
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
                    diffusion=FullDiffusion(
                        lambda x, u, t: jnp.eye(state_dim, bm_dim)
                    ),
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

    @implements(_infer_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        **kwargs,
    ) -> FunctionOfTime:
        if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
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
