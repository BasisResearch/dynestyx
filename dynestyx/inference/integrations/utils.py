from typing import Any

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints


class WeightedParticles(dist.Distribution):
    """A distribution over a finite set of weighted particles.

    Samples by drawing an index from a Categorical distribution defined by
    the log weights and returning the corresponding particle. Intended for
    representing particle filter posteriors as proper NumPyro distributions
    without relying on ``MixtureSameFamily``, which requires component
    distributions to have a parameter-free support constraint.

    Note:
        ``log_prob`` is not implemented because this distribution is intended
        solely for forward sampling (e.g., as an ``initial_condition`` in a
        ``DynamicalModel``). Using it as a likelihood or in contexts that require
        density evaluation will raise ``NotImplementedError``.

        ``arg_constraints`` is empty because ``particles`` and ``log_weights``
        are treated as fixed data (pre-computed particle filter output), not
        learnable distribution parameters.

    Args:
        particles: Array of shape ``(*batch_shape, n_particles, state_dim)``.
        log_weights: Array of shape ``(*batch_shape, n_particles)`` containing
            (possibly unnormalized) log weights.
    """

    arg_constraints: dict = {}
    support: Any = constraints.real_vector
    pytree_data_fields = ("particles", "log_weights")
    particles: jax.Array
    log_weights: jax.Array

    def __init__(
        self, particles: jax.Array, log_weights: jax.Array, validate_args=None
    ):
        if particles.ndim < 2:
            raise ValueError(
                "particles must have shape (*batch_shape, n_particles, state_dim)."
            )
        if log_weights.shape != particles.shape[:-1]:
            raise ValueError(
                "log_weights must have shape particles.shape[:-1]. "
                f"Got particles.shape={particles.shape}, log_weights.shape={log_weights.shape}."
            )

        self.particles = particles
        self.log_weights = log_weights
        batch_shape = particles.shape[:-2]
        event_shape = particles.shape[-1:]
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
            validate_args=validate_args,
        )

    def sample(self, key, sample_shape=()):
        idx = dist.Categorical(logits=self.log_weights).sample(key, sample_shape)
        n_particles = self.particles.shape[-2]
        state_dim = self.particles.shape[-1]

        particles = jnp.broadcast_to(
            self.particles,
            idx.shape + (n_particles, state_dim),
        )
        flat_particles = particles.reshape((-1, n_particles, state_dim))
        flat_idx = idx.reshape((-1,))
        sampled = jax.vmap(lambda p, i: p[i])(flat_particles, flat_idx)
        return sampled.reshape(idx.shape + (state_dim,))

    def log_prob(self, value):
        raise NotImplementedError("log_prob is not implemented for WeightedParticles.")


def particles_to_delta_mixtures(
    particles: jax.Array, log_weights: jax.Array
) -> list[dist.Distribution]:
    """Convert particles and weights to per-time weighted-particle distributions.

    Expects canonical shapes:
    - particles: (T, n_particles, state_dim)
    - log_weights: (T, n_particles)
    """
    assert particles.ndim == 3, (
        "Expected particles with shape (T, n_particles, state_dim), "
        f"got shape={particles.shape}."
    )
    assert log_weights.ndim == 2, (
        "Expected log_weights with shape (T, n_particles), "
        f"got shape={log_weights.shape}."
    )
    assert particles.shape[:2] == log_weights.shape[:2], (
        "Expected particles.shape[:2] == log_weights.shape[:2], "
        f"got {particles.shape[:2]} and {log_weights.shape[:2]}."
    )

    z = jax.nn.log_softmax(log_weights, axis=-1)
    return [WeightedParticles(particles[i], z[i]) for i in range(particles.shape[0])]
