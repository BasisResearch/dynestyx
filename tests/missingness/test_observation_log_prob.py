import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from jaxtyping import TypeCheckError

from dynestyx.models import DynamicalModel, LinearGaussianObservation
from dynestyx.observation_missingness import (
    _masked_multivariate_normal_log_prob,
    prepare_missing_observation_metadata,
    prepare_observation_log_prob,
    prepare_observation_mask,
)
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


def test_observation_log_prob_init_tracks_partial_and_full_row_missingness():
    obs_values = jnp.array(
        [
            [1.0, 2.0],
            [jnp.nan, 3.0],
            [jnp.nan, jnp.nan],
        ]
    )

    (
        filled_obs,
        obs_mask,
        _,
        has_missing,
        has_partial_missing,
        has_fully_missing_rows,
        _,
    ) = prepare_observation_mask(obs_values)

    assert has_missing
    assert has_partial_missing
    assert has_fully_missing_rows
    assert jnp.array_equal(
        obs_mask,
        jnp.array([[True, True], [False, True], [False, False]]),
    )
    assert jnp.allclose(filled_obs[0], obs_values[0])
    assert jnp.allclose(filled_obs[1], jnp.array([0.0, 3.0]))


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


def test_observation_log_prob_scalar_rows_zero_out_full_missing_steps():
    obs_values = jnp.array([[jnp.nan], [1.25]])
    log_prob, _, _, _ = prepare_observation_log_prob(
        _build_scalar_dynamics(lambda x, u, t: dist.Normal(x + t, 0.4)),
        obs_values,
    )

    assert jnp.allclose(
        log_prob(x=jnp.array(0.2), u=None, t=jnp.array(0.0), t_idx=0),
        0.0,
    )
    assert jnp.allclose(
        log_prob(x=jnp.array(0.2), u=None, t=jnp.array(1.0), t_idx=1),
        dist.Normal(1.2, 0.4).log_prob(1.25),
    )


def test_observation_log_prob_accepts_scalar_time_series():
    obs_values = jnp.array([jnp.nan, 1.25])
    log_prob, _, _, _ = prepare_observation_log_prob(
        _build_scalar_dynamics(lambda x, u, t: dist.Normal(x + t, 0.4)),
        obs_values,
    )

    assert jnp.allclose(
        log_prob(x=jnp.array(0.2), u=None, t=jnp.array(0.0), t_idx=0),
        0.0,
    )
    assert jnp.allclose(
        log_prob(x=jnp.array(0.2), u=None, t=jnp.array(1.0), t_idx=1),
        dist.Normal(1.2, 0.4).log_prob(1.25),
    )


def test_observation_log_prob_rejects_more_than_two_dimensions():
    with pytest.raises(TypeCheckError, match="parameter 'obs_values'"):
        prepare_observation_log_prob(
            _build_scalar_dynamics(lambda x, u, t: dist.Normal(x + t, 0.4)),
            jnp.zeros((1, 1, 1)),
        )


def test_observation_log_prob_partial_missing_unsupported_distribution_raises_at_init():
    obs_values = jnp.array([[1.0, jnp.nan]])
    with pytest.raises(
        NotImplementedError,
        match="Partial missingness currently requires",
    ):
        prepare_observation_log_prob(
            _build_vector_dynamics(lambda x, u, t: dist.Delta(x, event_dim=1)),
            obs_values,
            missing_observation_strategy="marginalize",
        )


def test_observation_log_prob_partial_missing_guard_survives_jit():
    dynamics = _build_vector_dynamics(lambda x, u, t: dist.Delta(x, event_dim=1))

    def score(obs_values):
        log_prob, _, _, _ = prepare_observation_log_prob(dynamics, obs_values)
        return log_prob(
            x=jnp.zeros(2),
            u=None,
            t=jnp.array(0.0),
            t_idx=0,
        )

    with pytest.raises(Exception, match="Partial missingness currently requires"):
        jax.block_until_ready(jax.jit(score)(jnp.array([[0.0, jnp.nan]])))


def test_observation_log_prob_partial_missing_type_change_raises_clear_error():
    obs_values = jnp.array([[1.0, jnp.nan], [1.0, jnp.nan]])

    def observation_model(x, u, t):
        if float(t) < 0.5:
            return dist.MultivariateNormal(x, covariance_matrix=GAUSSIAN_R)
        return dist.Delta(x, event_dim=1)

    log_prob, _, _, _ = prepare_observation_log_prob(
        _build_vector_dynamics(observation_model),
        obs_values,
    )

    with pytest.raises(
        ValueError,
        match="Partial missingness requires a time-stable marginalizable observation family",
    ):
        log_prob(
            x=jnp.array([1.0, 2.0]),
            u=None,
            t=jnp.array(1.0),
            t_idx=1,
        )


def test_observation_log_prob_linear_gaussian_matches_manual_reference():
    obs_values = jnp.array([[jnp.nan, 0.2]])
    log_prob, _, _, _ = prepare_observation_log_prob(
        _build_vector_dynamics(LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R)),
        obs_values,
    )
    x = jnp.array([0.5, -0.3])
    actual = log_prob(
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


def test_observation_log_prob_augment_student_t_matches_completed_data_reference():
    scale_tril = jnp.array([[0.4, 0.0], [0.15, 0.5]])
    obs_values = jnp.array([[1.0, jnp.nan]])
    dynamics = _build_vector_dynamics(
        lambda x, u, t: dist.MultivariateStudentT(
            df=5.0,
            loc=x,
            scale_tril=scale_tril,
        )
    )
    metadata = prepare_missing_observation_metadata(
        dynamics,
        obs_times=jnp.array([0.0]),
        obs_values=obs_values,
    )
    log_prob, completed, missing_times, coordinate_indices = (
        prepare_observation_log_prob(
            dynamics,
            obs_values,
            obs_times=jnp.array([0.0]),
            missing_observation_strategy="augment",
            missing_obs_values=jnp.array([0.3]),
            missing_obs_metadata=metadata,
        )
    )
    x = jnp.array([0.5, -0.2])
    completed_obs = jnp.array([1.0, 0.3])
    actual = log_prob(x=x, u=None, t=jnp.array(0.0), t_idx=0)
    expected = dist.MultivariateStudentT(
        df=5.0,
        loc=x,
        scale_tril=scale_tril,
    ).log_prob(completed_obs)

    assert completed is not None
    assert missing_times is not None
    assert coordinate_indices is not None
    assert jnp.allclose(completed[0], completed_obs)
    assert jnp.array_equal(missing_times, jnp.array([0.0]))
    assert jnp.array_equal(
        coordinate_indices,
        jnp.array([1], dtype=jnp.int32),
    )
    assert jnp.allclose(actual, expected)


def test_observation_log_prob_scalar_augmentation_preserves_scalar_shape():
    obs_times = jnp.array([0.0, 1.0])
    obs_values = jnp.array([1.0, jnp.nan])
    dynamics = _build_scalar_dynamics(lambda x, u, t: dist.Normal(x + t, 0.4))
    metadata = prepare_missing_observation_metadata(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
    )
    log_prob, completed, missing_times, coordinate_indices = (
        prepare_observation_log_prob(
            dynamics,
            obs_values,
            obs_times=obs_times,
            missing_observation_strategy="augment",
            missing_obs_values=jnp.array([0.3]),
            missing_obs_metadata=metadata,
        )
    )

    assert completed is not None
    assert missing_times is not None
    assert completed.shape == obs_values.shape
    assert jnp.allclose(completed, jnp.array([1.0, 0.3]))
    assert jnp.array_equal(missing_times, jnp.array([1.0]))
    assert coordinate_indices is None
    assert jnp.allclose(
        log_prob(x=jnp.array(0.2), u=None, t=jnp.array(1.0), t_idx=1),
        dist.Normal(1.2, 0.4).log_prob(0.3),
    )
