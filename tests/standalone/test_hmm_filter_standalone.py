import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from numpyro.handlers import trace

import dynestyx as dsx
from dynestyx.inference.filter_configs import HMMConfig
from dynestyx.inference.filters import Filter
from dynestyx.inference.hmm_filters import compute_hmm_filter
from dynestyx.models import DynamicalModel
from dynestyx.types import ConditionedResult
from tests.missingness.models import GAUSSIAN_R
from tests.missingness.utils import set_full_row_missing, set_partial_row_missing

HMM_TRANSITION = jnp.array([[0.82, 0.18], [0.27, 0.73]])
HMM_INITIAL_PROBS = jnp.array([0.55, 0.45])
HMM_OBS_LOC = jnp.array(
    [
        [-0.9, 0.4],
        [1.1, -0.6],
    ]
)


def _build_hmm_dynamics(observation_model):
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.Categorical(probs=HMM_INITIAL_PROBS),
        state_evolution=lambda x, u, t_now, t_next: dist.Categorical(
            probs=HMM_TRANSITION[x]
        ),
        observation_model=observation_model,
    )


def _mvn_observation_model(x, u, t):
    return dist.MultivariateNormal(HMM_OBS_LOC[x], covariance_matrix=GAUSSIAN_R)


def _independent_categorical_observation_model(x, u, t):
    probs = jnp.array(
        [
            [[0.85, 0.15], [0.7, 0.3]],
            [[0.2, 0.8], [0.35, 0.65]],
        ]
    )
    return dist.Independent(dist.Categorical(probs=probs[x]), 1)


def _joint_categorical_observation_model(x, u, t):
    probs = jnp.array(
        [
            [0.62, 0.18, 0.12, 0.08],
            [0.08, 0.14, 0.18, 0.60],
        ]
    )
    return dist.Categorical(probs=probs[x])


def _condition_hmm(*, obs_times, obs_values, observation_model):
    with Filter(
        filter_config=HMMConfig(record_filtered=True, record_log_filtered=True)
    ):
        return dsx.condition(
            "f",
            _build_hmm_dynamics(observation_model),
            obs_times=obs_times,
            obs_values=obs_values,
        )


def _plated_condition_hmm(*, obs_times, obs_values, observation_model, M):
    dynamics = _build_hmm_dynamics(observation_model)
    with Filter(
        filter_config=HMMConfig(record_filtered=True, record_log_filtered=True)
    ):
        with dsx.plate("trajectories", M):
            return dsx.condition(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
            )


def test_condition_hmm_full_row_missing_returns_finite_normalized_probs():
    obs_times = jnp.arange(6.0)
    obs_values = jnp.array(
        [
            [-0.7, 0.4],
            [-0.3, 0.2],
            [0.9, -0.4],
            [1.2, -0.7],
            [0.8, -0.2],
            [-0.5, 0.1],
        ]
    )
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_full_row_missing(obs_values, 3)

    result = _condition_hmm(
        obs_times=obs_times,
        obs_values=obs_values,
        observation_model=_mvn_observation_model,
    )

    assert isinstance(result, ConditionedResult)
    assert jnp.isfinite(result.marginal_loglik)
    assert result.states.shape == (len(obs_times), 2)
    assert len(result.dists) == len(obs_times)
    assert jnp.allclose(jnp.exp(result.states).sum(axis=-1), 1.0)


def test_condition_hmm_does_not_register_numpyro_sites_under_missingness():
    obs_times = jnp.arange(4.0)
    obs_values = jnp.array(
        [
            [-0.8, 0.3],
            [jnp.nan, jnp.nan],
            [1.0, -0.5],
            [0.7, -0.3],
        ]
    )

    with trace() as tr:
        result = _condition_hmm(
            obs_times=obs_times,
            obs_values=obs_values,
            observation_model=_mvn_observation_model,
        )

    assert isinstance(result, ConditionedResult)
    assert "f_marginal_loglik" not in tr
    assert "f_filtered_states" not in tr
    assert "f_log_filtered_states" not in tr


def test_condition_plated_hmm_gaussian_missingness_matches_memberwise_filter():
    M = 2
    obs_times = jnp.arange(5.0)
    obs_values = jnp.array(
        [
            [
                [-0.8, 0.3],
                [0.2, -0.2],
                [1.0, -0.5],
                [0.7, -0.3],
                [-0.6, 0.2],
            ],
            [
                [-0.9, 0.4],
                [1.2, -0.6],
                [0.1, 0.3],
                [0.9, -0.2],
                [1.1, -0.4],
            ],
        ]
    )
    obs_values = set_full_row_missing(obs_values, 1, member_idx=0)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0, member_idx=1)

    result = _plated_condition_hmm(
        obs_times=obs_times,
        obs_values=obs_values,
        observation_model=_mvn_observation_model,
        M=M,
    )

    expected = jnp.stack(
        [
            compute_hmm_filter(
                _build_hmm_dynamics(_mvn_observation_model),
                obs_times=obs_times,
                obs_values=obs_values[member_idx],
            )[1]
            for member_idx in range(M)
        ],
        axis=0,
    )

    assert isinstance(result, ConditionedResult)
    assert result.states.shape == (M, len(obs_times), 2)
    assert jnp.allclose(result.states, expected)


def test_condition_plated_hmm_independent_categorical_missingness_matches_memberwise_filter():
    M = 2
    obs_times = jnp.arange(4.0)
    obs_values = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        ]
    )
    obs_values = set_partial_row_missing(obs_values, 1, dim_idx=0, member_idx=0)
    obs_values = set_full_row_missing(obs_values, 2, member_idx=1)

    result = _plated_condition_hmm(
        obs_times=obs_times,
        obs_values=obs_values,
        observation_model=_independent_categorical_observation_model,
        M=M,
    )

    expected = jnp.stack(
        [
            compute_hmm_filter(
                _build_hmm_dynamics(_independent_categorical_observation_model),
                obs_times=obs_times,
                obs_values=obs_values[member_idx],
            )[1]
            for member_idx in range(M)
        ],
        axis=0,
    )

    assert isinstance(result, ConditionedResult)
    assert result.states.shape == (M, len(obs_times), 2)
    assert jnp.allclose(result.states, expected)


def test_condition_hmm_scalar_categorical_integer_labels_match_float_path():
    obs_times = jnp.arange(5.0)
    obs_values_int = jnp.array([0, 1, 3, 2, 1], dtype=jnp.int32)
    obs_values_float = obs_values_int.astype(jnp.float32)

    result_int = _condition_hmm(
        obs_times=obs_times,
        obs_values=obs_values_int,
        observation_model=_joint_categorical_observation_model,
    )
    result_float = _condition_hmm(
        obs_times=obs_times,
        obs_values=obs_values_float,
        observation_model=_joint_categorical_observation_model,
    )

    assert jnp.isfinite(result_int.marginal_loglik)
    assert jnp.allclose(result_int.marginal_loglik, result_float.marginal_loglik)
    assert jnp.allclose(result_int.states, result_float.states)


@pytest.mark.parametrize(
    ("obs_values", "message"),
    [
        (
            jnp.array([0.0, -1.0, 2.0]),
            "zero-based integer labels 0..K-1; found negative observed value",
        ),
        (
            jnp.array([0.0, 1.5, 2.0]),
            "zero-based integer labels. Found non-integer observed value",
        ),
        (
            jnp.array([0.0, 4.0, 2.0]),
            "zero-based integer labels 0..K-1.*with K=4",
        ),
    ],
)
def test_condition_hmm_categorical_observations_reject_invalid_labels_early(
    obs_values,
    message,
):
    obs_times = jnp.arange(3.0)

    with pytest.raises(ValueError, match=message):
        _condition_hmm(
            obs_times=obs_times,
            obs_values=obs_values,
            observation_model=_joint_categorical_observation_model,
        )
