import math
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jaxtyping import Array, Float, PRNGKeyArray, Real, Shaped
from numpyro.distributions import constraints

from dynestyx.distributions import RaoBlackwellizedParticleDistribution
from dynestyx.inference.integrations.utils import (
    WeightedParticles,
    covariance_from_cholesky,
)
from dynestyx.inference.utils.plate_utils import (
    _slice_time_axis,
    _time_len_from_array,
)

MissingPolicy = Literal["raise", "empty"]


class _ForwardSimulationImproperUniform(dist.ImproperUniform):
    """An improper distribution sampled by dynamical forward simulation.

    The forward sampler is used only for initialization and forward execution.
    Density evaluation is inherited from ``ImproperUniform`` and is always
    zero; it does not evaluate the log density of the ``DynamicalModel``.
    """

    pytree_data_fields = ("forward_sampler", "sample_transform")
    has_rsample = True

    def __init__(
        self,
        forward_sampler: Callable[[PRNGKeyArray], Shaped[Array, "..."]]
        | dist.Distribution,
        *,
        event_shape: tuple[int, ...],
        sample_transform: Callable[[Shaped[Array, "..."]], Shaped[Array, "..."]]
        | None = None,
        validate_args: bool | None = None,
    ) -> None:
        self.forward_sampler = forward_sampler
        self.sample_transform = sample_transform
        super().__init__(
            constraints.real,
            batch_shape=(),
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def _sample_one(self, key: PRNGKeyArray) -> Shaped[Array, "..."]:
        sample = (
            self.forward_sampler.sample(key)
            if isinstance(self.forward_sampler, dist.Distribution)
            else self.forward_sampler(key)
        )
        if self.sample_transform is not None:
            sample = self.sample_transform(sample)
        return jnp.reshape(jnp.asarray(sample), self.event_shape)

    def sample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Shaped[Array, "..."]:
        if not sample_shape:
            return self._sample_one(key)

        n_samples = math.prod(sample_shape)
        keys = jax.random.split(key, n_samples)
        samples = jax.vmap(self._sample_one)(keys)
        return jnp.reshape(samples, sample_shape + self.event_shape)

    def rsample(
        self,
        key: PRNGKeyArray,
        sample_shape: tuple[int, ...] = (),
    ) -> Shaped[Array, "..."]:
        return self.sample(key, sample_shape=sample_shape)


def _handle_missing_gaussian_sequence(
    *,
    missing: MissingPolicy,
    missing_message: str | None,
) -> list[dist.Distribution]:
    if missing == "empty":
        return []
    if missing == "raise":
        raise ValueError(
            missing_message or "Gaussian means/covariances were unavailable."
        )
    raise ValueError(f"Unknown missing Gaussian sequence policy: {missing!r}.")


def _gaussian_sequence_to_dists(
    means: Float[Array, "*plate time state_dim"] | None,
    covariances: Float[Array, "*plate time state_dim state_dim"] | None,
    *,
    plate_shapes: tuple[int, ...] = (),
    missing: MissingPolicy = "raise",
    missing_message: str | None = None,
) -> list[dist.Distribution]:
    """Convert time-indexed Gaussian parameters to per-time distributions."""
    if means is None or covariances is None:
        return _handle_missing_gaussian_sequence(
            missing=missing,
            missing_message=missing_message,
        )

    t_len = _time_len_from_array(means, plate_shapes)
    return [
        dist.MultivariateNormal(
            _slice_time_axis(means, t, plate_shapes),
            covariance_matrix=_slice_time_axis(covariances, t, plate_shapes),
        )
        for t in range(t_len)
    ]


def _particle_sequence_to_dists(
    particles: Real[Array, "*plate time n_particles state_dim"]
    | Real[Array, "*plate time n_particles"],
    log_weights: Float[Array, "*plate time n_particles"],
    *,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    """Convert time-indexed particle arrays to per-time weighted particles."""
    if particles.ndim == len(plate_shapes) + 2:
        particles = particles[..., None]

    normalized_log_weights = jax.nn.log_softmax(log_weights, axis=-1)
    t_len = _time_len_from_array(normalized_log_weights, plate_shapes)
    return [
        WeightedParticles(
            particles=_slice_time_axis(particles, t, plate_shapes),
            log_weights=_slice_time_axis(normalized_log_weights, t, plate_shapes),
        )
        for t in range(t_len)
    ]


def _posterior_sequence_to_dists(
    posterior,
    *,
    means_attr: str,
    covariances_attr: str,
    particle_mode: bool,
    plate_shapes: tuple[int, ...] = (),
    missing: MissingPolicy = "raise",
    missing_message: str | None = None,
) -> list[dist.Distribution]:
    """Convert a backend posterior object to per-time distributions."""
    if particle_mode:
        return _particle_sequence_to_dists(
            posterior.particles,
            posterior.log_weights,
            plate_shapes=plate_shapes,
        )

    return _gaussian_sequence_to_dists(
        getattr(posterior, means_attr),
        getattr(posterior, covariances_attr),
        plate_shapes=plate_shapes,
        missing=missing,
        missing_message=missing_message,
    )


def _rbpf_sequence_to_dists(
    posterior,
    *,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    """Convert an RBPF Gaussian-mixture sequence to joint-state distributions."""
    t_len = _time_len_from_array(posterior.weights, plate_shapes)
    return [
        RaoBlackwellizedParticleDistribution(
            log_weights=jnp.log(_slice_time_axis(posterior.weights, t, plate_shapes)),
            regimes=_slice_time_axis(posterior.regimes, t, plate_shapes),
            continuous_locs=_slice_time_axis(posterior.means, t, plate_shapes),
            continuous_covariances=_slice_time_axis(
                posterior.covariances, t, plate_shapes
            ),
        )
        for t in range(t_len)
    ]


def _cholesky_state_sequence_to_dists(
    states,
    *,
    particle_mode: bool,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    """Convert cuthbert state objects to per-time distributions."""
    if particle_mode:
        return _particle_sequence_to_dists(
            states.particles,
            states.log_weights,
            plate_shapes=plate_shapes,
        )

    return _gaussian_sequence_to_dists(
        states.mean,
        covariance_from_cholesky(states.chol_cov),
        plate_shapes=plate_shapes,
    )


def _categorical_log_probs_to_dists(
    log_probs: Float[Array, "*plate time n_states"],
    *,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    """Convert time-indexed categorical log-probs to per-time distributions."""
    t_len = _time_len_from_array(log_probs, plate_shapes)
    return [
        dist.Categorical(probs=jnp.exp(_slice_time_axis(log_probs, t, plate_shapes)))
        for t in range(t_len)
    ]
