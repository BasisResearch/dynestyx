"""Observed marginals through the public distribution scorer."""

import math

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

import dynestyx as dsx

COVARIANCE = jnp.array([[0.5, 0.1, 0.08], [0.1, 0.4, 0.05], [0.08, 0.05, 0.3]])
OBSERVED = jnp.array([0, 2])


@pytest.mark.parametrize(
    ("mean_batch", "covariance_batch"),
    [((2, 3), (2, 3)), ((2, 3), ()), ((), (2, 3)), ((2, 1), (1, 3))],
    ids=["aligned", "shared_covariance", "shared_mean", "crossed_axes"],
)
@pytest.mark.parametrize(
    "observed",
    [OBSERVED, jnp.arange(3), jnp.array([], dtype=int)],
    ids=["partial", "complete", "missing"],
)
def test_masked_gaussian_preserves_batch_axes_and_gradients(
    mean_batch, covariance_batch, observed
):
    loc = (
        jnp.arange(math.prod(mean_batch) * 3, dtype=float).reshape(mean_batch + (3,))
        / 20
    )
    covariance = COVARIANCE * jnp.linspace(
        0.7, 1.3, math.prod(covariance_batch)
    ).reshape(covariance_batch + (1, 1))
    mask = jnp.zeros(3, dtype=bool).at[observed].set(True)
    y = jnp.where(mask, jnp.array([0.4, 0.2, 0.7]), jnp.nan)

    def score(mean, cov):
        return dsx.masked_observation_log_prob(
            dist.MultivariateNormal(mean, cov),
            y=y,
            obs_mask=mask,
        )

    def reference(mean, cov):
        batch_shape = jnp.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
        if observed.size == 0:
            return jnp.zeros(batch_shape)
        mean = jnp.broadcast_to(mean, batch_shape + (3,))
        cov = jnp.broadcast_to(cov, batch_shape + (3, 3))
        return dist.MultivariateNormal(
            mean[..., observed], cov[..., observed[:, None], observed]
        ).log_prob(y[observed])

    actual = jax.jit(score)(loc, covariance)
    actual_gradients = jax.grad(
        lambda mean, cov: score(mean, cov).sum(), argnums=(0, 1)
    )(loc, covariance)
    expected_gradients = jax.grad(
        lambda mean, cov: reference(mean, cov).sum(), argnums=(0, 1)
    )(loc, covariance)

    assert actual.shape == (2, 3)
    assert jnp.allclose(actual, reference(loc, covariance))
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert jnp.all(jnp.isfinite(actual_gradient))
        assert jnp.allclose(actual_gradient, expected_gradient, atol=1e-6)


def test_masked_independent_preserves_batches_and_ignores_missing_values():
    loc = jnp.array([[0.1, -0.2, 0.3], [0.3, 0.4, -0.1]])
    y = jnp.array([0.4, jnp.nan, 0.7])

    def score(mean):
        return dsx.masked_observation_log_prob(
            dist.LogNormal(mean, 0.5).to_event(1),
            y=y,
            obs_mask=jnp.array([True, False, True]),
        )

    def reference(mean):
        return dist.LogNormal(mean[..., OBSERVED], 0.5).log_prob(y[OBSERVED]).sum(-1)

    actual = jax.jit(score)(loc)
    gradient = jax.grad(lambda mean: score(mean).sum())(loc)

    assert actual.shape == (2,)
    assert jnp.allclose(actual, reference(loc))
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, jax.grad(lambda mean: reference(mean).sum())(loc))


@pytest.mark.parametrize("observed", [False, True])
def test_masked_scalar_preserves_singleton_batch_axis(observed):
    observation_dist = dist.Normal(jnp.array([0.3]), 0.5)
    actual = jax.jit(dsx.masked_observation_log_prob)(
        observation_dist,
        y=jnp.array(0.9 if observed else jnp.nan),
        obs_mask=jnp.array(observed),
    )
    expected = observation_dist.log_prob(0.9) if observed else jnp.zeros(1)

    assert actual.shape == (1,)
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("observed", [False, True])
def test_other_vector_families_support_complete_or_missing_rows(observed):
    observation_dist = dist.MultivariateStudentT(5.0, jnp.zeros(2), jnp.eye(2))
    actual = jax.jit(dsx.masked_observation_log_prob)(
        observation_dist,
        y=jnp.full(2, 0.3 if observed else jnp.nan),
        obs_mask=jnp.full(2, observed),
    )
    expected = observation_dist.log_prob(jnp.full(2, 0.3)) if observed else 0.0

    assert jnp.allclose(actual, expected)


def test_unsupported_partial_marginalization_raises_eagerly_and_under_jit():
    observation_dist = dist.MultivariateStudentT(5.0, jnp.zeros(2), jnp.eye(2))

    def score(mask):
        return dsx.masked_observation_log_prob(
            observation_dist, y=jnp.array([0.3, jnp.nan]), obs_mask=mask
        )

    mask = jnp.array([True, False])
    with pytest.raises(ValueError, match="Partial missingness currently requires"):
        score(mask)
    with pytest.raises(Exception, match="Partial missingness currently requires"):
        jax.block_until_ready(jax.jit(score)(mask))


def test_observations_cannot_broadcast_across_event_axes():
    with pytest.raises(ValueError, match="observation event shape"):
        dsx.masked_observation_log_prob(
            dist.MultivariateNormal(jnp.zeros(3), COVARIANCE),
            y=jnp.zeros(1),
            obs_mask=jnp.ones(1, dtype=bool),
        )
