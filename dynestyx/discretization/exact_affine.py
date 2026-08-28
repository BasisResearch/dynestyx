"""Exact discretization of affine SDEs."""

from typing import ClassVar, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jaxtyping import Array, Real

from dynestyx.discretization.numerics import (
    _finalize_covariance,
    _positive_interval,
    _symmetrize,
)
from dynestyx.models import (
    AffineDrift,
    LinearGaussianParams,
    LinearGaussianStateEvolution,
    StochasticContinuousTimeStateEvolution,
)

_MAX_COVARIANCE_SQUARINGS = 16


def _van_loan_covariance(
    F: Real[Array, "state_dim state_dim"],
    diffusion_cov: Real[Array, "state_dim state_dim"],
    h: Real[Array, ""],
) -> Real[Array, "state_dim state_dim"]:
    """Evaluate the Van Loan covariance on a safe interval and compose it."""
    state_dim = F.shape[-1]
    interval_drift_norm = jnp.linalg.norm(F * h, ord=1)
    outer_squarings = jnp.ceil(
        jnp.log2(jnp.maximum(jnp.asarray(1.0, dtype=F.dtype), interval_drift_norm))
    ).astype(jnp.int32)
    h = eqx.error_if(
        h,
        outer_squarings > _MAX_COVARIANCE_SQUARINGS,
        "Discretization requires more than 16 covariance scaling squarings; "
        "reduce the interval length or drift magnitude.",
    )
    short_h = h / (2.0 ** outer_squarings.astype(F.dtype))

    van_loan = jnp.block(
        [
            [F, diffusion_cov],
            [jnp.zeros_like(F), -F.T],
        ]
    )
    short_exponential = jsp.linalg.expm(van_loan * short_h)
    short_A = short_exponential[:state_dim, :state_dim]
    short_Q = _symmetrize(short_exponential[:state_dim, state_dim:] @ short_A.T)

    def _double_interval(carry):
        A, Q = carry
        doubled_Q = _symmetrize(Q + A @ Q @ A.T)
        return A @ A, doubled_Q

    def _scan_step(carry, index):
        carry = jax.lax.cond(
            index < outer_squarings,
            _double_interval,
            lambda values: values,
            carry,
        )
        return carry, None

    (_, covariance), _ = jax.lax.scan(
        _scan_step,
        (short_A, short_Q),
        jnp.arange(_MAX_COVARIANCE_SQUARINGS),
    )
    return covariance


def _affine_transition_parameters(
    F: Real[Array, "state_dim state_dim"],
    B: Real[Array, "state_dim control_dim"] | None,
    b: Real[Array, " state_dim"] | None,
    L: Real[Array, "state_dim bm_dim"],
    h: Real[Array, ""],
    *,
    covariance_jitter: float,
) -> LinearGaussianParams:
    """Discretize a time-invariant affine SDE without matrix inverses."""
    F = jnp.asarray(F)
    state_dim = F.shape[-1]

    blocks: list[Array] = []
    if B is not None:
        blocks.append(jnp.asarray(B, dtype=F.dtype))
    if b is not None:
        blocks.append(jnp.asarray(b, dtype=F.dtype)[:, None])

    if blocks:
        forcing = jnp.concatenate(blocks, axis=-1)
        forcing_dim = forcing.shape[-1]
        augmented = jnp.block(
            [
                [F, forcing],
                [
                    jnp.zeros((forcing_dim, state_dim), dtype=F.dtype),
                    jnp.zeros((forcing_dim, forcing_dim), dtype=F.dtype),
                ],
            ]
        )
        exp_augmented = jsp.linalg.expm(augmented * h)
        A_h = exp_augmented[:state_dim, :state_dim]
        integrated_forcing = exp_augmented[:state_dim, state_dim:]
        cursor = 0
        if B is None:
            B_h = None
        else:
            control_dim = B.shape[-1]
            B_h = integrated_forcing[:, :control_dim]
            cursor = control_dim
        bias_h = None if b is None else integrated_forcing[:, cursor]
    else:
        A_h = jsp.linalg.expm(F * h)
        B_h = None
        bias_h = None

    L = jnp.asarray(L, dtype=F.dtype)
    diffusion_cov = L @ L.T
    Q_h = _van_loan_covariance(F, diffusion_cov, h)
    Q_h = _finalize_covariance(Q_h, covariance_jitter=covariance_jitter)
    return LinearGaussianParams(A=A_h, B=B_h, bias=bias_h, cov=Q_h)


def _exact_affine_parameters(
    cte: StochasticContinuousTimeStateEvolution,
    t_now,
    t_next,
    *,
    covariance_jitter: float,
) -> LinearGaussianParams:
    h = _positive_interval(t_now, t_next)
    drift = cte.drift
    assert isinstance(drift, AffineDrift)
    F = jnp.asarray(drift.A)
    L = jnp.asarray(
        cte.diffusion.as_matrix(
            x=None,
            u=None,
            t=0,
            state_dim=F.shape[-1],
        )
    )
    return _affine_transition_parameters(
        F,
        drift.B,
        drift.b,
        L,
        h,
        covariance_jitter=covariance_jitter,
    )


class _ExactAffineParameter(eqx.Module):
    cte: StochasticContinuousTimeStateEvolution
    name: Literal["A", "B", "bias", "cov"] = eqx.field(static=True)
    covariance_jitter: float = eqx.field(static=True)

    def __call__(self, t_now, t_next):
        return getattr(
            _exact_affine_parameters(
                self.cte,
                t_now,
                t_next,
                covariance_jitter=self.covariance_jitter,
            ),
            self.name,
        )


class _ExactAffineStateEvolution(LinearGaussianStateEvolution):
    """Exact affine transition selected by ``ExactAffineConfig``."""

    _dynestyx_discretizer_preserves_state_shape: ClassVar[bool] = True
    cte: StochasticContinuousTimeStateEvolution
    covariance_jitter: float = eqx.field(static=True)

    def __init__(
        self,
        cte: StochasticContinuousTimeStateEvolution,
        *,
        covariance_jitter: float,
    ):
        if not isinstance(cte.drift, AffineDrift):
            raise TypeError("ExactAffineConfig requires drift to be an AffineDrift.")
        if cte.potential is not None:
            raise TypeError("ExactAffineConfig does not support a potential term.")
        if callable(cte.diffusion.coefficient):
            raise TypeError(
                "ExactAffineConfig requires structurally constant additive diffusion."
            )

        self.cte = cte
        self.covariance_jitter = covariance_jitter
        drift = cte.drift
        super().__init__(
            A=_ExactAffineParameter(cte, "A", covariance_jitter),
            B=(
                None
                if drift.B is None
                else _ExactAffineParameter(cte, "B", covariance_jitter)
            ),
            bias=(
                None
                if drift.b is None
                else _ExactAffineParameter(cte, "bias", covariance_jitter)
            ),
            cov=_ExactAffineParameter(cte, "cov", covariance_jitter),
        )

    def params_at(self, t_now, t_next) -> LinearGaussianParams:
        return _exact_affine_parameters(
            self.cte,
            t_now,
            t_next,
            covariance_jitter=self.covariance_jitter,
        )
