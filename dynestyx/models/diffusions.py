"""Diffusion objects for continuous-time state evolution."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from dynestyx.types import Control, State, Time

DiffusionValue = Array | float | int
DiffusionSpec = Callable[[State, Control | None, Time], DiffusionValue] | DiffusionValue


class EvaluatedDiffusion(NamedTuple):
    """A diffusion coefficient evaluated at a specific ``(x, u, t)``.

    This is primarily a developer-facing helper used by solvers and backend
    integrations. It pairs a structured ``Diffusion`` object with the concrete
    value of its coefficient at one state, control, and time.
    """

    diffusion: Diffusion
    value: Array

    def as_matrix(self, *, state_dim: int) -> Array:
        """Return the evaluated diffusion coefficient as a matrix ``L``."""
        return self.diffusion._value_as_matrix(self.value, state_dim=state_dim)

    def gram_matrix(self, *, state_dim: int) -> Array:
        """L L^T."""
        return self.diffusion._value_gram_matrix(self.value, state_dim=state_dim)

    def apply(self, dw: Array, *, state_dim: int) -> Array:
        """Return ``L @ dw`` using the structured diffusion representation."""
        return self.diffusion._apply_value(self.value, dw, state_dim=state_dim)


class Diffusion(eqx.Module):
    """Base class for diffusion coefficients in SDEs.

    A diffusion encapsulates both the numeric coefficient ``L(x, u, t)`` and the
    structural interpretation of that coefficient inside the SDE

    ``dx_t = f(x_t, u_t, t) dt + L(x_t, u_t, t) dW_t``.

    Most users should instantiate one of the concrete subclasses:

    - ``FullDiffusion`` for an arbitrary matrix-valued coefficient.
    - ``DiagonalDiffusion`` for axis-aligned loadings.
    - ``ScalarDiffusion`` for isotropic or shared-driver noise.

    The ``coefficient`` may be either:

    - a constant array or scalar, for time-homogeneous diffusion, or
    - a callable ``(x, u, t) -> value`` for state-, control-, or time-dependent diffusion.

    ``bm_dim`` is the Brownian dimension ``d_w``. For ``FullDiffusion`` it can be
    inferred from the matrix shape. For ``DiagonalDiffusion`` and
    ``ScalarDiffusion`` it must be specified explicitly and must be either ``1`` or
    ``state_dim``.
    """

    coefficient: DiffusionSpec
    bm_dim: int | None = eqx.field(static=True, default=None)

    def __init__(
        self,
        coefficient: DiffusionSpec,
        bm_dim: int | None = None,
    ):
        self.coefficient = (
            coefficient if callable(coefficient) else jnp.asarray(coefficient)
        )
        self.bm_dim = None if bm_dim is None else int(bm_dim)
        self._validate_init()

    def evaluate_value(
        self,
        *,
        x: State,
        u: Control | None,
        t: Time,
    ) -> Array:
        """Return the raw coefficient value at ``(x, u, t)``."""
        if callable(self.coefficient):
            return jnp.asarray(self.coefficient(x, u, t))
        return cast(Array, self.coefficient)

    def resolve_metadata(
        self,
        *,
        state_dim: int,
        x_probe: State,
        u_probe: Control | None,
        t_probe: Time,
    ) -> Diffusion:
        """Validate coefficient shape information and resolve diffusion metadata."""
        raise NotImplementedError

    def evaluate(
        self,
        *,
        x: State,
        u: Control | None,
        t: Time,
    ) -> EvaluatedDiffusion:
        """Evaluate the diffusion at ``(x, u, t)``."""
        return EvaluatedDiffusion(self, self.evaluate_value(x=x, u=u, t=t))

    def as_matrix(
        self,
        *,
        x: State,
        u: Control | None,
        t: Time,
        state_dim: int,
    ) -> Array:
        """Return the matrix-valued diffusion coefficient ``L(x, u, t)``."""
        return self.evaluate(x=x, u=u, t=t).as_matrix(state_dim=state_dim)

    def gram_matrix(
        self,
        *,
        x: State,
        u: Control | None,
        t: Time,
        state_dim: int,
    ) -> Array:
        """Return the diffusion Gram matrix ``L(x,u,t) L(x,u,t)^T``."""
        return self.evaluate(x=x, u=u, t=t).gram_matrix(state_dim=state_dim)

    def apply(
        self,
        dw: Array,
        *,
        x: State,
        u: Control | None,
        t: Time,
        state_dim: int,
    ) -> Array:
        """Apply the diffusion coefficient to a Brownian increment ``dw``."""
        return self.evaluate(x=x, u=u, t=t).apply(dw, state_dim=state_dim)

    def _with_bm_dim(self, bm_dim: int) -> Diffusion:
        return type(self)(self.coefficient, bm_dim=int(bm_dim))

    def _constant_shape(self) -> tuple[int, ...] | None:
        if callable(self.coefficient):
            return None
        return tuple(int(d) for d in jnp.shape(self.coefficient))

    def _validate_init(self) -> None:
        if self.bm_dim is not None and int(self.bm_dim) <= 0:
            raise ValueError(f"bm_dim must be positive. Got bm_dim={self.bm_dim}.")

    def _value_as_matrix(self, value: Array, *, state_dim: int) -> Array:
        raise NotImplementedError("Please don't construct `Diffusion` directly; instead instantiate one of its subclasses (e.g., `FullDiffusion`, `DiagonalDiffusion`, or `ScalarDiffusion`)")

    def _value_gram_matrix(self, value: Array, *, state_dim: int) -> Array:
        raise NotImplementedError

    def _apply_value(self, value: Array, dw: Array, *, state_dim: int) -> Array:
        raise NotImplementedError


class FullDiffusion(Diffusion):
    """General matrix-valued diffusion coefficient.

    Use ``FullDiffusion`` when you want to specify the diffusion matrix
    ``L(x, u, t)`` directly.

    Args:
        coefficient: Either a constant array with trailing shape
            ``(..., state_dim, bm_dim)`` or a callable ``(x, u, t) -> array``
            with that trailing shape.
        bm_dim: Optional Brownian dimension. If omitted for a constant
            coefficient, it is inferred from the trailing matrix dimension.

    This is the most general public diffusion class. Prefer it when your model
    genuinely needs a dense or otherwise unstructured loading matrix.
    """

    def _validate_init(self) -> None:
        super()._validate_init()
        shape = self._constant_shape()
        if shape is not None and len(shape) < 2:
            raise ValueError(
                "Full diffusion requires a matrix-valued constant coefficient with "
                "trailing shape (..., state_dim, bm_dim). "
                f"Got shape {shape}."
            )
        if shape is not None and self.bm_dim is None:
            self.bm_dim = int(shape[-1])

    def resolve_metadata(
        self,
        *,
        state_dim: int,
        x_probe: State,
        u_probe: Control | None,
        t_probe: Time,
    ) -> FullDiffusion:
        shape = jax.eval_shape(
            lambda: self.evaluate_value(x=x_probe, u=u_probe, t=t_probe)
        ).shape
        if len(shape) < 2 or int(shape[-2]) != state_dim:
            raise ValueError(
                "Full diffusion must have trailing shape (..., state_dim, bm_dim). "
                f"Got shape {shape} with state_dim={state_dim}."
            )
        inferred_bm_dim = int(shape[-1])
        if self.bm_dim is not None and int(self.bm_dim) != inferred_bm_dim:
            raise ValueError(
                "bm_dim does not match inferred diffusion output shape. "
                f"Got bm_dim={self.bm_dim}, inferred={inferred_bm_dim} from shape {shape}."
            )
        return (
            self
            if self.bm_dim == inferred_bm_dim
            else cast(FullDiffusion, self._with_bm_dim(inferred_bm_dim))
        )

    def _value_as_matrix(self, value: Array, *, state_dim: int) -> Array:
        return value

    def _value_gram_matrix(self, value: Array, *, state_dim: int) -> Array:
        return value @ jnp.swapaxes(value, -1, -2)

    def _apply_value(self, value: Array, dw: Array, *, state_dim: int) -> Array:
        return value @ dw


class DiagonalDiffusion(Diffusion):
    """Vector-valued diffusion with diagonal or shared-driver interpretation.

    Use ``DiagonalDiffusion(v, bm_dim=...)`` when the diffusion is naturally
    parameterized by a vector ``v`` of length ``state_dim``.

    Args:
        coefficient: Either a constant vector with trailing shape
            ``(..., state_dim)`` or a callable ``(x, u, t) -> array`` with that
            trailing shape.
        bm_dim: Brownian dimension. Must be either ``1`` or ``state_dim``.

    - If ``bm_dim = state_dim``, the vector is interpreted as the diagonal of
      ``L = diag(v)``.
    - If ``bm_dim = 1``, the vector is interpreted as a column loading vector,
      so all state coordinates share one Brownian driver.

    This is often the right public API choice when each state coordinate has its
    own scale but you do not want to write out a full matrix.
    """

    def _validate_init(self) -> None:
        super()._validate_init()
        if self.bm_dim is None:
            raise ValueError(
                "Diagonal diffusion requires explicit bm_dim. "
                "For diagonal diffusion, bm_dim must be either 1 or state_dim."
            )
        shape = self._constant_shape()
        if shape is not None and len(shape) == 0:
            raise ValueError(
                "Diagonal diffusion requires a vector-valued constant coefficient "
                "with trailing shape (..., state_dim). "
                f"Got shape {shape}."
            )

    def resolve_metadata(
        self,
        *,
        state_dim: int,
        x_probe: State,
        u_probe: Control | None,
        t_probe: Time,
    ) -> DiagonalDiffusion:
        shape = jax.eval_shape(
            lambda: self.evaluate_value(x=x_probe, u=u_probe, t=t_probe)
        ).shape
        if len(shape) == 0 or int(shape[-1]) != state_dim:
            raise ValueError(
                "Diagonal diffusion must have trailing shape (..., state_dim). "
                f"Got shape {shape} with state_dim={state_dim}."
            )
        bm_dim = self.bm_dim
        assert bm_dim is not None
        if bm_dim not in (1, state_dim):
            raise ValueError(
                "Diagonal diffusion requires bm_dim to be either 1 or state_dim. "
                f"Got bm_dim={bm_dim}, state_dim={state_dim}."
            )
        return self

    def _value_as_matrix(self, value: Array, *, state_dim: int) -> Array:
        assert self.bm_dim is not None
        if self.bm_dim == 1:
            return value[..., :, None]
        return value[..., :, None] * jnp.eye(state_dim, dtype=value.dtype)

    def _value_gram_matrix(self, value: Array, *, state_dim: int) -> Array:
        assert self.bm_dim is not None
        if self.bm_dim == 1:
            return value[..., :, None] * value[..., None, :]
        return jnp.square(value)[..., :, None] * jnp.eye(state_dim, dtype=value.dtype)

    def _apply_value(self, value: Array, dw: Array, *, state_dim: int) -> Array:
        assert self.bm_dim is not None
        if self.bm_dim == 1:
            return value * dw[..., 0]
        return value * dw


class ScalarDiffusion(Diffusion):
    """Scalar-valued diffusion with isotropic or shared-driver interpretation.

    Use ``ScalarDiffusion(sigma, bm_dim=...)`` when the diffusion is controlled
    by a single scalar scale ``sigma``.

    Args:
        coefficient: Either a scalar, a constant array with trailing shape
            ``(..., 1)``, or a callable ``(x, u, t) -> scalar_or_length_1_array``.
        bm_dim: Brownian dimension. Must be either ``1`` or ``state_dim``.

    - If ``bm_dim = state_dim``, this represents isotropic diffusion
      ``L = sigma I``.
    - If ``bm_dim = 1``, this represents a shared scalar driver applied equally
      to every state coordinate.

    This is typically the simplest public API choice for constant isotropic
    diffusion, and is usually preferable to writing ``sigma * eye(state_dim)``
    by hand.
    """

    def _validate_init(self) -> None:
        super()._validate_init()
        if self.bm_dim is None:
            raise ValueError(
                "Scalar diffusion requires explicit bm_dim. "
                "For scalar diffusion, bm_dim must be either 1 or state_dim."
            )
        shape = self._constant_shape()
        if shape is not None and len(shape) != 0 and int(shape[-1]) != 1:
            raise ValueError(
                "Scalar diffusion requires a scalar constant coefficient or trailing "
                "shape (..., 1). "
                f"Got shape {shape}."
            )

    def resolve_metadata(
        self,
        *,
        state_dim: int,
        x_probe: State,
        u_probe: Control | None,
        t_probe: Time,
    ) -> ScalarDiffusion:
        shape = jax.eval_shape(
            lambda: self.evaluate_value(x=x_probe, u=u_probe, t=t_probe)
        ).shape
        if len(shape) != 0 and int(shape[-1]) != 1:
            raise ValueError(
                "Scalar diffusion must have shape () or trailing shape (..., 1). "
                f"Got shape {shape}."
            )
        bm_dim = self.bm_dim
        assert bm_dim is not None
        if bm_dim not in (1, state_dim):
            raise ValueError(
                "Scalar diffusion requires bm_dim to be either 1 or state_dim. "
                f"Got bm_dim={bm_dim}, state_dim={state_dim}."
            )
        return self

    def _scalar_value(self, value: Array) -> Array:
        return value if value.ndim == 0 else jnp.squeeze(value, axis=-1)

    def _value_as_matrix(self, value: Array, *, state_dim: int) -> Array:
        scalar = self._scalar_value(value)
        assert self.bm_dim is not None
        if self.bm_dim == 1:
            return jnp.broadcast_to(
                scalar[..., None, None], scalar.shape + (state_dim, 1)
            )
        return scalar[..., None, None] * jnp.eye(state_dim, dtype=value.dtype)

    def _value_gram_matrix(self, value: Array, *, state_dim: int) -> Array:
        sigma_sq = jnp.square(self._scalar_value(value))
        assert self.bm_dim is not None
        if self.bm_dim == 1:
            return sigma_sq[..., None, None] * jnp.ones(
                (state_dim, state_dim), dtype=value.dtype
            )
        return sigma_sq[..., None, None] * jnp.eye(state_dim, dtype=value.dtype)

    def _apply_value(self, value: Array, dw: Array, *, state_dim: int) -> Array:
        scalar = self._scalar_value(value)
        assert self.bm_dim is not None
        if self.bm_dim == 1:
            return jnp.broadcast_to(
                (scalar * dw[..., 0])[..., None], scalar.shape + (state_dim,)
            )
        return scalar[..., None] * dw
