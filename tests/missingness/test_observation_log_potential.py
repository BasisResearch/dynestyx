import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from dynestyx.internal.observation_missingness import (
    ObservationLogPotential,
    _masked_multivariate_normal_log_prob,
)
from dynestyx.models import DynamicalModel, LinearGaussianObservation
from tests.missingness.models import GAUSSIAN_R, INDEPENDENT_SCALE
from tests.missingness.utils import (
    manual_masked_independent_normal_log_prob,
    manual_masked_mvn_log_prob,
)


def _build_vector_dynamics(observation_model):
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            x, jnp.eye(2)
        ),
        observation_model=observation_model,
        control_dim=0,
    )


def _build_scalar_dynamics(observation_model):
    return DynamicalModel(
        initial_condition=dist.Normal(0.0, 1.0),
        state_evolution=lambda x, u, t_now, t_next: dist.Normal(x, 1.0),
        observation_model=observation_model,
        control_dim=0,
    )


def test_observation_log_potential_init_tracks_partial_and_full_row_missingness():
    obs_values = jnp.array(
        [
            [1.0, 2.0],
            [jnp.nan, 3.0],
            [jnp.nan, jnp.nan],
        ]
    )

    log_potential = ObservationLogPotential(
        dynamics=_build_vector_dynamics(
            lambda x, u, t: dist.MultivariateNormal(
                jnp.zeros(2), covariance_matrix=jnp.eye(2)
            )
        ),
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


def test_masked_independent_distribution_matches_manual_subset_formula():
    loc = jnp.array([0.4, -0.7])
    y = jnp.array([1.2, 0.0])
    obs_mask = jnp.array([True, False])
    obs_dist = dist.Independent(dist.Normal(loc, INDEPENDENT_SCALE), 1)

    actual = obs_dist.base_dist.mask(obs_mask).to_event(1).log_prob(y)
    expected = manual_masked_independent_normal_log_prob(
        loc, INDEPENDENT_SCALE, y, obs_mask
    )

    assert jnp.allclose(actual, expected)


def test_observation_log_potential_scalar_rows_zero_out_full_missing_steps():
    obs_values = jnp.array([jnp.nan, 1.25])
    log_potential = ObservationLogPotential(
        dynamics=_build_scalar_dynamics(lambda x, u, t: dist.Normal(x + t, 0.4)),
        obs_values=obs_values,
    )

    assert jnp.allclose(
        log_potential.log_potential_step(
            x=jnp.array(0.2), u=None, t=jnp.array(0.0), t_idx=0
        ),
        0.0,
    )
    assert jnp.allclose(
        log_potential.log_potential_step(
            x=jnp.array(0.2), u=None, t=jnp.array(1.0), t_idx=1
        ),
        dist.Normal(1.2, 0.4).log_prob(1.25),
    )


def test_observation_log_potential_partial_missing_unsupported_distribution_raises_at_init():
    obs_values = jnp.array([[1.0, jnp.nan]])
    with pytest.raises(
        NotImplementedError,
        match="Partial missingness currently requires",
    ):
        ObservationLogPotential(
            dynamics=_build_vector_dynamics(lambda x, u, t: dist.Delta(x, event_dim=1)),
            obs_values=obs_values,
        )


def test_observation_log_potential_can_fail_late_if_distribution_type_changes():
    obs_values = jnp.array([[1.0, jnp.nan], [1.0, jnp.nan]])

    def observation_model(x, u, t):
        if float(t) < 0.5:
            return dist.MultivariateNormal(x, covariance_matrix=GAUSSIAN_R)
        return dist.Delta(x, event_dim=1)

    log_potential = ObservationLogPotential(
        dynamics=_build_vector_dynamics(observation_model),
        obs_values=obs_values,
    )

    with pytest.raises(Exception, match="obs_dist"):
        log_potential.log_potential_step(
            x=jnp.array([1.0, 2.0]),
            u=None,
            t=jnp.array(1.0),
            t_idx=1,
        )


def test_observation_log_potential_linear_gaussian_matches_manual_reference():
    obs_values = jnp.array([[jnp.nan, 0.2]])
    log_potential = ObservationLogPotential(
        dynamics=_build_vector_dynamics(
            LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R)
        ),
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
