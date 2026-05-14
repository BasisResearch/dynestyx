from typing import Literal

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from dynestyx.inference.integrations.utils import (
    WeightedParticles,
    covariance_from_cholesky,
)
from dynestyx.inference.plate_utils import _slice_time_axis, _time_len_from_array

MissingPolicy = Literal["raise", "empty"]


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
    means: jax.Array | None,
    covariances: jax.Array | None,
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
    particles: jax.Array,
    log_weights: jax.Array,
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
    log_probs: jax.Array,
    *,
    plate_shapes: tuple[int, ...] = (),
) -> list[dist.Distribution]:
    """Convert time-indexed categorical log-probs to per-time distributions."""
    t_len = _time_len_from_array(log_probs, plate_shapes)
    return [
        dist.Categorical(probs=jnp.exp(_slice_time_axis(log_probs, t, plate_shapes)))
        for t in range(t_len)
    ]
