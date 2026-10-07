"""Structured, sliceable covariance specifications for Gaussian models."""

from abc import abstractmethod
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import Float, Real

from dynestyx.utils.arrays import _real_array
from dynestyx.utils.validation import _raise_now_or_error_if, _validate_array


def _resolve_variance(
    *,
    sd: Real[ArrayLike, "..."] | None,
    variance: Real[ArrayLike, "..."] | None,
) -> Float[Array, "..."]:
    """Read exactly one finite, nonnegative SD or variance input and return variances."""
    if (sd is None) == (variance is None):
        raise ValueError("Specify exactly one of sd or variance.")
    selected = sd if sd is not None else variance
    assert selected is not None
    value = _real_array(selected)
    value = _raise_now_or_error_if(
        value,
        jnp.any(~jnp.isfinite(value) | (value < 0)),
        "Standard deviations and variances must be finite and nonnegative.",
    )
    variance_value = jnp.square(value) if sd is not None else value
    return _raise_now_or_error_if(
        variance_value,
        jnp.any(~jnp.isfinite(variance_value)),
        "Variances must be finite.",
    )


class Covariance(eqx.Module):
    """Base interface for scalar, diagonal, and full covariance specifications.

    Numeric fields are JAX pytree leaves. Leading axes are batch/plate axes;
    ``event_rank`` identifies the trailing axes belonging to one covariance.
    """

    event_rank: ClassVar[int]

    @property
    @abstractmethod
    def value(self) -> Float[Array, "..."]:
        """Return the numeric leaf with ``event_rank`` trailing covariance axes."""
        raise NotImplementedError

    @abstractmethod
    def as_matrix(
        self, event_dim: int
    ) -> Float[Array, "*batch {event_dim} {event_dim}"]:
        """Return a dense covariance with trailing shape ``(event_dim, event_dim)``."""
        raise NotImplementedError


class ScalarCovariance(Covariance):
    """Isotropic covariance ``variance * I``; specify ``sd`` or ``variance``.

    An array of scalars represents leading batch/plate dimensions.
    """

    event_rank: ClassVar[int] = 0
    variance: Float[Array, "*batch"]

    def __init__(
        self,
        *,
        sd: Real[ArrayLike, "*batch"] | None = None,
        variance: Real[ArrayLike, "*batch"] | None = None,
    ) -> None:
        """Treat every input axis as a batch axis of isotropic variances."""
        self.variance = _resolve_variance(sd=sd, variance=variance)

    @property
    def value(self) -> Float[Array, "*batch"]:
        """Return one variance per batch member, with no trailing event axes."""
        return self.variance

    def as_matrix(
        self, event_dim: int
    ) -> Float[Array, "*batch {event_dim} {event_dim}"]:
        """Expand each scalar variance to an isotropic ``event_dim`` covariance."""
        return self.variance[..., None, None] * jnp.eye(
            event_dim, dtype=self.variance.dtype
        )


class DiagonalCovariance(Covariance):
    """Diagonal covariance; the trailing axis holds variances or standard deviations."""

    event_rank: ClassVar[int] = 1
    variance: Float[Array, "*batch event_dim"]

    def __init__(
        self,
        *,
        sd: Real[ArrayLike, "..."] | None = None,
        variance: Real[ArrayLike, "..."] | None = None,
    ) -> None:
        """Use the trailing input axis for event variances and leading axes for batches."""
        self.variance = _resolve_variance(sd=sd, variance=variance)
        if self.variance.ndim < 1 or self.variance.shape[-1] == 0:
            raise ValueError(
                "DiagonalCovariance requires a nonempty trailing vector axis."
            )

    @property
    def value(self) -> Float[Array, "*batch event_dim"]:
        """Return the diagonal variances, retaining their trailing event axis."""
        return self.variance

    def as_matrix(
        self, event_dim: int
    ) -> Float[Array, "*batch {event_dim} {event_dim}"]:
        """Build diagonal matrices after checking the trailing event dimension."""
        if self.variance.shape[-1] != event_dim:
            raise ValueError(
                f"Diagonal covariance dimension must be {event_dim}; got {self.variance.shape}."
            )
        return self.variance[..., :, None] * jnp.eye(
            event_dim, dtype=self.variance.dtype
        )


class FullCovariance(Covariance):
    """A full covariance matrix with two trailing event axes.

    Leading axes represent batches/plates. Construction checks shape, finite
    values, and symmetry. Supply a positive-semidefinite matrix, with a
    positive-definite effective covariance when a Gaussian density is needed.
    """

    event_rank: ClassVar[int] = 2
    matrix: Float[Array, "*batch event_dim event_dim"]

    def __init__(self, matrix: Real[ArrayLike, "..."]) -> None:
        """Validate square trailing event axes, finite values, and symmetry."""
        self.matrix = _real_array(matrix)
        if (
            self.matrix.ndim < 2
            or self.matrix.shape[-1] == 0
            or self.matrix.shape[-2] != self.matrix.shape[-1]
        ):
            raise ValueError(
                "FullCovariance requires nonempty square trailing matrix axes."
            )
        self.matrix = _validate_array(
            self.matrix, name="Covariance", symmetric=True, atol=1e-7
        )

    @property
    def value(self) -> Float[Array, "*batch event_dim event_dim"]:
        """Return the dense matrix with two trailing covariance event axes."""
        return self.matrix

    def as_matrix(
        self, event_dim: int
    ) -> Float[Array, "*batch {event_dim} {event_dim}"]:
        """Return the matrix after checking both trailing event dimensions."""
        if self.matrix.shape[-2:] != (event_dim, event_dim):
            raise ValueError(
                f"Full covariance dimension must be {event_dim}; got {self.matrix.shape}."
            )
        return self.matrix


def covariance_matrix(
    value: Covariance | Real[ArrayLike, "*batch event_dim event_dim"],
    event_dim: int,
) -> Real[Array, "*batch event_dim event_dim"]:
    """Return a dense covariance with shape ``(*batch, event_dim, event_dim)``.

    Scalar and diagonal covariance objects expand using ``event_dim``. Full
    objects check their event dimension. Existing dense arrays retain their
    values and dtype; leading batch/plate axes are preserved in either case.
    """
    return (
        value.as_matrix(event_dim)
        if isinstance(value, Covariance)
        else jnp.asarray(value)
    )


def construct_gaussian(
    loc: Real[Array, "..."],
    covariance: Covariance | Real[ArrayLike, "*batch event_dim event_dim"],
) -> dist.Normal | dist.MultivariateNormal:
    """Construct a Normal for scalar loc, otherwise a MultivariateNormal.

    For array loc, the trailing axis is the event axis; leading axes are batches.
    """
    dim = 1 if loc.ndim == 0 else loc.shape[-1]
    matrix = covariance_matrix(covariance, dim)
    if loc.ndim == 0:
        return dist.Normal(loc, jnp.sqrt(matrix[..., 0, 0]))
    return dist.MultivariateNormal(loc=loc, covariance_matrix=matrix)
