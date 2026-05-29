import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from dynestyx.internal.observation_missingness import (
    MissingObservationLogPotential,
    _masked_independent_log_prob,
    _masked_multivariate_normal_log_prob,
)
from dynestyx.models import LinearGaussianObservation
from tests.missingness.models import GAUSSIAN_R, INDEPENDENT_SCALE
from tests.missingness.utils import (
    manual_masked_independent_normal_log_prob,
    manual_masked_mvn_log_prob,
)


def test_missing_observation_log_potential_init_tracks_partial_and_full_row_missingness():
    obs_values = jnp.array(
        [
            [1.0, 2.0],
            [jnp.nan, 3.0],
            [jnp.nan, jnp.nan],
        ]
    )

    log_potential = MissingObservationLogPotential(
        observation_model=lambda x, u, t: dist.Normal(0.0, 1.0),
        obs_values=obs_values,
    )

    assert log_potential.has_missing
    assert log_potential.has_partial_missing
    assert log_potential.has_fully_missing_rows
    assert jnp.array_equal(
        log_potential.obs_mask,
        jnp.array([[True, True], [False, True], [False, False]]),
    )
    assert jnp.allclose(log_potential.safe_obs[0], obs_values[0])
    assert jnp.allclose(log_potential.safe_obs[1], jnp.array([0.0, 3.0]))


def test_masked_multivariate_normal_log_prob_matches_manual_subset_formula():
    mu = jnp.array([0.3, -0.2])
    y = jnp.array([1.0, 0.0])
    obs_mask = jnp.array([True, False])
    obs_dist = dist.MultivariateNormal(mu, covariance_matrix=GAUSSIAN_R)

    actual = _masked_multivariate_normal_log_prob(obs_dist, y, obs_mask)
    expected = manual_masked_mvn_log_prob(mu, GAUSSIAN_R, y, obs_mask)

    assert jnp.allclose(actual, expected)


def test_masked_independent_log_prob_matches_manual_subset_formula():
    loc = jnp.array([0.4, -0.7])
    y = jnp.array([1.2, 0.0])
    obs_mask = jnp.array([True, False])
    obs_dist = dist.Independent(dist.Normal(loc, INDEPENDENT_SCALE), 1)

    actual = _masked_independent_log_prob(obs_dist, y, obs_mask)
    expected = manual_masked_independent_normal_log_prob(
        loc, INDEPENDENT_SCALE, y, obs_mask
    )

    assert jnp.allclose(actual, expected)


def test_missing_observation_log_potential_scalar_rows_zero_out_full_missing_steps():
    obs_values = jnp.array([jnp.nan, 1.25])
    log_potential = MissingObservationLogPotential(
        observation_model=lambda x, u, t: dist.Normal(x + t, 0.4),
        obs_values=obs_values,
    )

    assert jnp.allclose(
        log_potential.log_potential_step(
            x=jnp.array(0.2),
            u=None,
            t=jnp.array(0.0),
            t_idx=0,
        ),
        0.0,
    )
    assert jnp.allclose(
        log_potential.log_potential_step(
            x=jnp.array(0.2),
            u=None,
            t=jnp.array(1.0),
            t_idx=1,
        ),
        dist.Normal(1.2, 0.4).log_prob(1.25),
    )


def test_missing_observation_log_potential_partial_missing_unsupported_distribution_raises():
    obs_values = jnp.array([[1.0, jnp.nan]])
    log_potential = MissingObservationLogPotential(
        observation_model=lambda x, u, t: dist.Delta(x, event_dim=1),
        obs_values=obs_values,
    )

    with pytest.raises(
        NotImplementedError,
        match="Partial missingness is currently supported only",
    ):
        log_potential.log_potential_step(
            x=jnp.array([1.0, 2.0]),
            u=None,
            t=jnp.array(0.0),
            t_idx=0,
        )


def test_missing_observation_log_potential_linear_gaussian_matches_manual_reference():
    obs_values = jnp.array([[jnp.nan, 0.2]])
    log_potential = MissingObservationLogPotential(
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
        obs_values=obs_values,
    )
    x = jnp.array([0.5, -0.3])
    actual = log_potential.log_potential_step(
        x=x,
        u=None,
        t=jnp.array(0.0),
        t_idx=0,
    )
    expected = manual_masked_mvn_log_prob(
        x,
        GAUSSIAN_R,
        jnp.array([0.0, 0.2]),
        jnp.array([False, True]),
    )
    assert jnp.allclose(actual, expected)
