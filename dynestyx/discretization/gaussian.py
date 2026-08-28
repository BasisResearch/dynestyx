"""Gaussian approximations for nonlinear SDE transitions."""

from typing import ClassVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Real

from dynestyx.discretization.exact_affine import _affine_transition_parameters
from dynestyx.discretization.numerics import (
    _finalize_covariance,
    _positive_interval,
    _symmetrize,
)
from dynestyx.inference.configs.discretizer import (
    BaseDiscretizerConfig,
    EulerMaruyamaConfig,
    LocalLinearizationConfig,
    MeanTrajectoryLinearizationConfig,
)
from dynestyx.models import (
    DiscreteTimeStateEvolution,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.solvers import euler_maruyama_loc_cov

type _Moments = tuple[
    Real[Array, " state_dim"],
    Real[Array, "state_dim state_dim"],
]


def _local_linearization_moments(
    cte: StochasticContinuousTimeStateEvolution,
    x,
    u,
    t_now,
    t_next,
    *,
    covariance_jitter: float,
) -> _Moments:
    h = _positive_interval(t_now, t_next)
    f0 = cte.total_drift(x=x, u=u, t=t_now)
    J = jax.jacfwd(lambda z: cte.total_drift(x=z, u=u, t=t_now))(x)
    L = cte.diffusion.as_matrix(
        x=None,
        u=None,
        t=0,
        state_dim=x.shape[-1],
    )
    params = _affine_transition_parameters(
        J,
        None,
        f0 - J @ x,
        L,
        h,
        covariance_jitter=covariance_jitter,
    )
    assert params.bias is not None
    return params.A @ x + params.bias, params.cov


def _mean_trajectory_moments(
    cte: StochasticContinuousTimeStateEvolution,
    x,
    u,
    t_now,
    t_next,
    *,
    ode_solver,
    covariance_jitter: float,
) -> _Moments:
    x = jnp.asarray(x)
    t_now = jnp.asarray(t_now, dtype=x.dtype)
    t_next = t_now + _positive_interval(t_now, t_next)
    state_dim = x.shape[-1]

    def _rhs(t, y, args):
        del args
        m, P = y
        P = _symmetrize(P)
        mean_drift = cte.total_drift(x=m, u=u, t=t)
        J = jax.jacfwd(lambda z: cte.total_drift(x=z, u=u, t=t))(m)
        diffusion_cov = cte.diffusion.gram_matrix(
            x=m,
            u=u,
            t=t,
            state_dim=state_dim,
        )
        covariance_derivative = J @ P + P @ J.T + diffusion_cov
        return mean_drift, _symmetrize(covariance_derivative)

    solution = dfx.diffeqsolve(
        dfx.ODETerm(_rhs),
        t0=t_now,
        t1=t_next,
        y0=(x, jnp.zeros((state_dim, state_dim), dtype=x.dtype)),
        saveat=dfx.SaveAt(t1=True),
        **ode_solver.diffeqsolve_settings,
    )
    return solution.ys[0][0], _finalize_covariance(
        solution.ys[1][0],
        covariance_jitter=covariance_jitter,
    )


def _configured_moments(
    cte: StochasticContinuousTimeStateEvolution,
    config: BaseDiscretizerConfig,
    x,
    u,
    t_now,
    t_next,
) -> _Moments:
    if isinstance(config, EulerMaruyamaConfig):
        t_now_arr = jnp.asarray(t_now)
        t_next_arr = jnp.asarray(t_next)
        _positive_interval(t_now_arr, t_next_arr)
        result = euler_maruyama_loc_cov(cte, x, u, t_now_arr, t_next_arr)
        return result["loc"], _finalize_covariance(
            result["cov"],
            covariance_jitter=config.covariance_jitter,
        )
    if isinstance(config, LocalLinearizationConfig):
        return _local_linearization_moments(
            cte,
            x,
            u,
            t_now,
            t_next,
            covariance_jitter=config.covariance_jitter,
        )
    if isinstance(config, MeanTrajectoryLinearizationConfig):
        return _mean_trajectory_moments(
            cte,
            x,
            u,
            t_now,
            t_next,
            ode_solver=config.ode_solver,
            covariance_jitter=config.covariance_jitter,
        )
    raise TypeError(
        f"Config does not define Gaussian moments: {type(config).__name__}."
    )


class _ConfiguredGaussianStateEvolution(DiscreteTimeStateEvolution):
    """Gaussian transition selected by a discretizer config."""

    _dynestyx_discretizer_preserves_state_shape: ClassVar[bool] = True
    cte: StochasticContinuousTimeStateEvolution
    config: BaseDiscretizerConfig = eqx.field(static=True)

    def __init__(
        self,
        cte: StochasticContinuousTimeStateEvolution,
        config: BaseDiscretizerConfig,
    ):
        self.cte = cte
        self.config = config

    def __call__(self, x, u, t_now, t_next):
        loc, covariance = _configured_moments(
            self.cte,
            self.config,
            x,
            u,
            t_now,
            t_next,
        )
        return dist.MultivariateNormal(loc=loc, covariance_matrix=covariance)
