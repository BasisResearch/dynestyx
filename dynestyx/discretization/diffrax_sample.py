"""Sample-only Diffrax transitions for one SDE interval."""

import math
from collections.abc import Callable
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jaxtyping import Array, PRNGKeyArray, Real
from numpyro.distributions import constraints

from dynestyx.inference.configs.discretizer import DiffraxSampleConfig
from dynestyx.models import (
    DiscreteTimeStateEvolution,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.solvers import solve_diffrax_sde_interval

_SAMPLE_ONLY_MESSAGE = (
    "DiffraxSampleConfig provides sampling only. It is supported by "
    "DiscreteTimeSimulator, cuthbert EnKF, cuthbert bootstrap PF, and "
    "genealogy-tracing particle smoothing, but not by algorithms that require "
    "transition densities or analytic moments."
)


class _DiffraxTransitionDistribution(dist.Distribution):
    """NumPyro-compatible, sample-only distribution for one SDE interval."""

    arg_constraints: dict[str, constraints.Constraint] = {}
    support = constraints.real_vector
    has_rsample = True
    reparametrized_params: list[str] = []

    def __init__(
        self,
        *,
        cte: StochasticContinuousTimeStateEvolution,
        sample_one: Callable[..., Array],
        x: Real[Array, " state_dim"],
        u,
        t_now,
        t_next,
        validate_args: bool | None = None,
    ):
        self.cte = cte
        self._sample_one = sample_one
        self.x = jnp.asarray(x)
        self.u = u
        self.t_now = jnp.asarray(t_now, dtype=self.x.dtype)
        self.t_next = jnp.asarray(t_next, dtype=self.x.dtype)
        super().__init__(
            batch_shape=(),
            event_shape=self.x.shape,
            validate_args=validate_args,
        )

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        if key is None:
            raise ValueError("Diffrax sample-only transitions require a PRNG key.")
        sample_shape = tuple(sample_shape)
        if not sample_shape:
            return self._sample_one(
                self.cte,
                self.x,
                self.u,
                self.t_now,
                self.t_next,
                key,
            )

        sample_count = math.prod(sample_shape)
        keys = jr.split(key, sample_count)
        samples = jax.vmap(
            lambda sample_key: self._sample_one(
                self.cte,
                self.x,
                self.u,
                self.t_now,
                self.t_next,
                sample_key,
            )
        )(keys)
        return jnp.reshape(samples, sample_shape + self.event_shape)

    def rsample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        return self.sample(key, sample_shape=sample_shape)

    def log_prob(self, value):
        del value
        raise NotImplementedError(_SAMPLE_ONLY_MESSAGE)

    @property
    def mean(self):
        raise NotImplementedError(_SAMPLE_ONLY_MESSAGE)

    @property
    def variance(self):
        raise NotImplementedError(_SAMPLE_ONLY_MESSAGE)


class _DiffraxSampleStateEvolution(DiscreteTimeStateEvolution):
    """Sample-only state evolution selected by ``DiffraxSampleConfig``."""

    _dynestyx_discretizer_preserves_state_shape: ClassVar[bool] = True
    cte: StochasticContinuousTimeStateEvolution
    sample_one: Callable[..., Array] = eqx.field(static=True)

    def __init__(
        self,
        cte: StochasticContinuousTimeStateEvolution,
        config: DiffraxSampleConfig,
    ):
        settings = config.sde_solver.diffeqsolve_settings
        tol_vbt = config.sde_solver.resolved_tol_vbt
        assert tol_vbt is not None

        def _sample_one(active_cte, x, u, t_now, t_next, key):
            return solve_diffrax_sde_interval(
                active_cte,
                initial_state=x,
                t0=t_now,
                t1=t_next,
                u=u,
                diffeqsolve_settings=settings,
                key=key,
                tol_vbt=tol_vbt,
            )

        self.cte = cte
        self.sample_one = _sample_one

    def __call__(self, x, u, t_now, t_next):
        return _DiffraxTransitionDistribution(
            cte=self.cte,
            sample_one=self.sample_one,
            x=x,
            u=u,
            t_now=t_now,
            t_next=t_next,
        )
