from types import SimpleNamespace

import jax
import jax.numpy as jnp

from dynestyx.inference.distribution_utils import (
    _categorical_log_probs_to_dists,
    _cholesky_state_sequence_to_dists,
    _gaussian_sequence_to_dists,
    _particle_sequence_to_dists,
    _posterior_sequence_to_dists,
)


def test_gaussian_sequence_to_dists_unbatched():
    means = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    covs = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))

    dists = _gaussian_sequence_to_dists(means, covs)

    assert len(dists) == 2
    assert dists[0].batch_shape == ()
    assert dists[0].event_shape == (2,)
    assert jnp.allclose(dists[1].loc, jnp.array([2.0, 3.0]))


def test_gaussian_sequence_to_dists_plate_batched():
    means = jnp.arange(12.0).reshape(2, 3, 2)
    covs = jnp.broadcast_to(jnp.eye(2), (2, 3, 2, 2))

    dists = _gaussian_sequence_to_dists(means, covs, plate_shapes=(2,))

    assert len(dists) == 3
    assert dists[0].batch_shape == (2,)
    assert dists[0].event_shape == (2,)
    assert dists[0].loc.shape == (2, 2)


def test_particle_sequence_to_dists_normalizes_and_plate_batches():
    particles = jnp.arange(48.0).reshape(2, 3, 4, 2)
    log_weights = jnp.zeros((2, 3, 4))

    dists = _particle_sequence_to_dists(
        particles,
        log_weights,
        plate_shapes=(2,),
    )

    assert len(dists) == 3
    assert dists[0].batch_shape == (2,)
    assert dists[0].event_shape == (2,)
    assert dists[0].particles.shape == (2, 4, 2)
    assert jnp.allclose(jnp.exp(dists[0].log_weights).sum(axis=-1), jnp.ones(2))


def test_cholesky_state_sequence_to_dists_gaussian():
    states = SimpleNamespace(
        mean=jnp.array([[0.0, 1.0], [2.0, 3.0]]),
        chol_cov=jnp.broadcast_to(2.0 * jnp.eye(2), (2, 2, 2)),
    )

    dists = _cholesky_state_sequence_to_dists(states, particle_mode=False)

    assert len(dists) == 2
    assert jnp.allclose(dists[0].covariance_matrix, 4.0 * jnp.eye(2))


def test_categorical_log_probs_to_dists_plate_batched():
    logits = jnp.arange(24.0).reshape(2, 3, 4)
    log_probs = jax.nn.log_softmax(logits, axis=-1)

    dists = _categorical_log_probs_to_dists(log_probs, plate_shapes=(2,))

    assert len(dists) == 3
    assert dists[0].batch_shape == (2,)
    assert dists[0].probs.shape == (2, 4)
    assert jnp.allclose(dists[0].probs.sum(axis=-1), jnp.ones(2))


def test_posterior_sequence_to_dists_uses_attrs_and_missing_empty():
    posterior = SimpleNamespace(
        filtered_means=jnp.array([[0.0], [1.0]]),
        filtered_covariances=jnp.ones((2, 1, 1)),
    )

    dists = _posterior_sequence_to_dists(
        posterior,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=False,
    )

    assert len(dists) == 2
    assert dists[1].loc.shape == (1,)

    missing_posterior = SimpleNamespace(
        filtered_means=None,
        filtered_covariances=None,
    )
    assert (
        _posterior_sequence_to_dists(
            missing_posterior,
            means_attr="filtered_means",
            covariances_attr="filtered_covariances",
            particle_mode=False,
            missing="empty",
        )
        == []
    )
