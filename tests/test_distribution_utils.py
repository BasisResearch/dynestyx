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
    """A full-rank ensemble keeps the dense `MultivariateNormal`, and its `log_prob`.

    `LowRankMultivariateNormal` divides by `cov_diag` to form its Woodbury
    capacitance factor, so with the default zero jitter its `log_prob` is `nan`
    even when the factor has full rank. Switching representation there would be a
    silent regression for every model with `n_particles > state_dim` and buys
    nothing: the dense covariance is well posed and at most `n_particles` wide.
    """
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


def test_cholesky_state_sequence_to_dists_ensemble_jitter_enables_log_prob():
    """The ensemble covariance is singular, so a density needs explicit jitter."""
    ensemble = jr.normal(jr.PRNGKey(0), (2, 4, 16))
    states = SimpleNamespace(ensemble=ensemble)

    without = _cholesky_state_sequence_to_dists(states, particle_mode=False)[0]
    assert jnp.isnan(without.log_prob(without.mean))

    with_jitter = _cholesky_state_sequence_to_dists(
        states, particle_mode=False, covariance_jitter=1e-2
    )[0]
    assert jnp.isfinite(with_jitter.log_prob(with_jitter.mean))
    assert jnp.allclose(
        with_jitter.covariance_matrix,
        without.covariance_matrix + 1e-2 * jnp.eye(16),
        atol=1e-5,
    )


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
