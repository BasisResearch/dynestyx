import jax
import numpyro.distributions as dist


def particles_to_delta_mixtures(
    particles: jax.Array, log_weights: jax.Array
) -> list[dist.Distribution]:
    """Convert particles and weights to per-time delta-mixture distributions.

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

    log_weights_norm = log_weights - jax.scipy.special.logsumexp(
        log_weights, axis=-1, keepdims=True
    )
    return [
        dist.MixtureSameFamily(
            dist.Categorical(logits=log_weights_norm[i]),
            dist.Delta(particles[i], event_dim=1),  # type: ignore[arg-type]
        )
        for i in range(particles.shape[0])
    ]
