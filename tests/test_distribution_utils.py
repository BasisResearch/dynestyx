from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist

from dynestyx.inference.utils.distribution_utils import (
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


def test_cholesky_state_sequence_to_dists_ensemble_is_low_rank():
    """Ensemble states keep the rank-(N-1) factor instead of a dense covariance.

    Expanding it would be both quadratic in ``state_dim`` and singular, and
    ``MultivariateNormal`` takes its Cholesky eagerly, so the dense path yields
    ``nan`` for ``sample`` and ``log_prob``. ``state_dim`` is deliberately larger
    than ``n_particles`` here: that is the regime the dense path gets wrong, and
    the one every other EnKF test misses by using ``state_dim=2``.
    """
    t_len, n_particles, state_dim = 3, 4, 16
    ensemble = jr.normal(jr.PRNGKey(0), (t_len, n_particles, state_dim))
    states = SimpleNamespace(ensemble=ensemble)

    dists = _cholesky_state_sequence_to_dists(states, particle_mode=False)

    assert len(dists) == t_len
    for t, d in enumerate(dists):
        assert isinstance(d, dist.LowRankMultivariateNormal)
        assert d.batch_shape == ()
        assert d.event_shape == (state_dim,)
        assert d.cov_factor.shape == (state_dim, n_particles)

        members = ensemble[t]
        assert jnp.allclose(d.mean, members.mean(axis=0), atol=1e-5)

        deviations = members - members.mean(axis=0)
        expected_cov = deviations.T @ deviations / (n_particles - 1)
        assert jnp.allclose(d.covariance_matrix, expected_cov, atol=1e-5)

        # The point of the change: the dense path samples nan here.
        assert jnp.isfinite(d.sample(jr.PRNGKey(t))).all()


def test_cholesky_state_sequence_to_dists_full_rank_ensemble_stays_dense():
    """A full-rank ensemble keeps the dense `MultivariateNormal`, and its `log_prob`."""
    n_particles, state_dim = 16, 4  # n_particles - 1 >= state_dim: full rank
    ensemble = jr.normal(jr.PRNGKey(0), (2, n_particles, state_dim))
    deviations = ensemble - ensemble.mean(axis=-2, keepdims=True)
    chol_cov = jnp.linalg.cholesky(
        jnp.einsum("tni,tnj->tij", deviations, deviations) / (n_particles - 1)
    )
    states = SimpleNamespace(
        ensemble=ensemble, mean=ensemble.mean(axis=-2), chol_cov=chol_cov
    )

    dists = _cholesky_state_sequence_to_dists(states, particle_mode=False)

    assert isinstance(dists[0], dist.MultivariateNormal)
    assert jnp.isfinite(dists[0].log_prob(dists[0].mean))


def test_covariance_jitter_shifts_only_the_covariance_diagonal():
    """The jitter adds exactly ``eps * I`` to the covariance and nothing else.

    Checked on both Gaussian branches, since they apply it by different means:
    the dense branch adds ``eps * I`` to the covariance directly, while the
    low-rank branch passes ``eps`` as `LowRankMultivariateNormal`'s ``cov_diag``
    and never forms the covariance at all. The mean must be untouched either way
    -- the jitter regularizes the reported covariance so a singular one gains a
    density, it is not a change of location.
    """
    jitter = 1e-5

    # Dense branch: a square Cholesky factor. Covariance is 2 * I @ (2 * I).T = 4 * I.
    dense_states = SimpleNamespace(
        mean=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        chol_cov=jnp.broadcast_to(2.0 * jnp.eye(2), (2, 2, 2)),
    )
    exact = _cholesky_state_sequence_to_dists(
        dense_states, particle_mode=False, covariance_jitter=0.0
    )[0]
    jittered = _cholesky_state_sequence_to_dists(
        dense_states, particle_mode=False, covariance_jitter=jitter
    )[0]

    assert isinstance(exact, dist.MultivariateNormal)
    assert jnp.array_equal(exact.covariance_matrix, 4.0 * jnp.eye(2))
    assert jnp.allclose(
        jittered.covariance_matrix,
        exact.covariance_matrix + jitter * jnp.eye(2),
        atol=1e-8,
    )
    assert jnp.array_equal(jittered.mean, exact.mean)

    # Low-rank branch: the ensemble covariance is
    # singular and only the jitter gives it a density.
    state_dim = 16
    ensemble_states = SimpleNamespace(
        ensemble=jr.normal(jr.PRNGKey(0), (2, 4, state_dim))
    )
    lr_exact = _cholesky_state_sequence_to_dists(
        ensemble_states, particle_mode=False, covariance_jitter=0.0
    )[0]
    lr_jittered = _cholesky_state_sequence_to_dists(
        ensemble_states, particle_mode=False, covariance_jitter=jitter
    )[0]

    assert isinstance(lr_exact, dist.LowRankMultivariateNormal)
    assert jnp.allclose(
        lr_jittered.covariance_matrix,
        lr_exact.covariance_matrix + jitter * jnp.eye(state_dim),
        atol=1e-6,
    )
    assert jnp.array_equal(lr_jittered.mean, lr_exact.mean)
    # The factor itself is untouched; the jitter lives entirely in cov_diag.
    assert jnp.array_equal(lr_jittered.cov_factor, lr_exact.cov_factor)
    assert jnp.allclose(lr_jittered.cov_diag, jnp.full((state_dim,), jitter))
    # Only the jittered one has a density.
    assert jnp.isnan(lr_exact.log_prob(lr_exact.mean))
    assert jnp.isfinite(lr_jittered.log_prob(lr_jittered.mean))


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
