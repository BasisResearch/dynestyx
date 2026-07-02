import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import trace
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Filter
from dynestyx.inference.filter_configs import HMMConfig
from dynestyx.inference.hmm_filters import compute_hmm_filter, hmm_log_components
from dynestyx.models import DynamicalModel
from tests.missingness.models import GAUSSIAN_R, INDEPENDENT_SCALE
from tests.missingness.utils import (
    manual_masked_independent_normal_log_prob,
    manual_masked_mvn_log_prob,
    set_full_row_missing,
    set_partial_row_missing,
)

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


def _independent_observation_model(x, u, t):
    return dist.Independent(
        dist.Normal(HMM_OBS_LOC[x], INDEPENDENT_SCALE),
        1,
    )


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


def _delta_observation_model(x, u, t):
    return dist.Delta(HMM_OBS_LOC[x], event_dim=1)


def _scalar_categorical_hmm_model(
    A=None,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    A = numpyro.sample(
        "A",
        dist.Dirichlet(jnp.ones(2)).expand([2]).to_event(1),
        obs=A,
    )

    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.Categorical(probs=jnp.ones(2) / 2),
        state_evolution=lambda x, u, t_now, t_next: dist.Categorical(probs=A[x]),
        observation_model=_joint_categorical_observation_model,
    )

    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def _run_hmm_filter_trace(dynamics, obs_times, obs_values):
    with trace() as tr:
        with Filter(
            filter_config=HMMConfig(record_filtered=True, record_log_filtered=True)
        ):
            dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)
    return tr


def _run_plated_hmm_filter_trace(model, *, obs_times, obs_values, M=2):
    with trace() as tr:
        with Filter(
            filter_config=HMMConfig(record_filtered=True, record_log_filtered=True)
        ):
            model(obs_times=obs_times, obs_values=obs_values, M=M)
    return tr


def _plated_hmm_model(observation_model, *, obs_times=None, obs_values=None, M=2):
    dynamics = _build_hmm_dynamics(observation_model)
    with dsx.plate("trajectories", M):
        dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)


def test_hmm_full_row_missing_rows_zero_emission_scores():
    obs_times = jnp.arange(5.0)
    obs_values = jnp.array(
        [
            [-0.8, 0.3],
            [0.2, -0.2],
            [1.0, -0.5],
            [0.7, -0.3],
            [-0.6, 0.2],
        ]
    )
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_full_row_missing(obs_values, 3)

    _, _, log_emit_seq = hmm_log_components(
        _build_hmm_dynamics(_mvn_observation_model),
        obs_times,
        obs_values,
    )

    assert log_emit_seq.shape == (len(obs_times), 2)
    assert jnp.allclose(log_emit_seq[1], jnp.zeros(2))
    assert jnp.allclose(log_emit_seq[3], jnp.zeros(2))


def test_hmm_full_row_missing_block_keeps_filter_finite_and_normalized():
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

    tr = _run_hmm_filter_trace(
        _build_hmm_dynamics(_mvn_observation_model),
        obs_times,
        obs_values,
    )

    filtered = tr["f_filtered_states"]["value"]
    log_filtered = tr["f_log_filtered_states"]["value"]

    assert filtered.shape == (len(obs_times), 2)
    assert log_filtered.shape == (len(obs_times), 2)
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
    assert jnp.allclose(filtered.sum(axis=-1), 1.0)


def test_hmm_partial_missing_independent_matches_manual_reference():
    obs_times = jnp.arange(2.0)
    obs_values = jnp.array(
        [
            [-0.4, 0.1],
            [1.2, -0.5],
        ]
    )
    obs_values = set_partial_row_missing(obs_values, 0, dim_idx=1)

    _, _, log_emit_seq = hmm_log_components(
        _build_hmm_dynamics(_independent_observation_model),
        obs_times,
        obs_values,
    )

    obs_mask = jnp.array([True, False])
    safe_obs = jnp.array([-0.4, 0.0])
    expected = jnp.array(
        [
            manual_masked_independent_normal_log_prob(
                HMM_OBS_LOC[0], INDEPENDENT_SCALE, safe_obs, obs_mask
            ),
            manual_masked_independent_normal_log_prob(
                HMM_OBS_LOC[1], INDEPENDENT_SCALE, safe_obs, obs_mask
            ),
        ]
    )

    assert jnp.allclose(log_emit_seq[0], expected)


def test_hmm_partial_missing_multivariate_normal_matches_manual_reference():
    obs_times = jnp.arange(2.0)
    obs_values = jnp.array(
        [
            [-0.4, 0.1],
            [1.2, -0.5],
        ]
    )
    obs_values = set_partial_row_missing(obs_values, 0, dim_idx=0)

    _, _, log_emit_seq = hmm_log_components(
        _build_hmm_dynamics(_mvn_observation_model),
        obs_times,
        obs_values,
    )

    obs_mask = jnp.array([False, True])
    safe_obs = jnp.array([0.0, 0.1])
    expected = jnp.array(
        [
            manual_masked_mvn_log_prob(HMM_OBS_LOC[0], GAUSSIAN_R, safe_obs, obs_mask),
            manual_masked_mvn_log_prob(HMM_OBS_LOC[1], GAUSSIAN_R, safe_obs, obs_mask),
        ]
    )

    assert jnp.allclose(log_emit_seq[0], expected)


def test_hmm_full_row_missing_works_for_non_marginalizable_observation_family():
    obs_times = jnp.arange(4.0)
    obs_values = jnp.array(
        [
            [-0.9, 0.4],
            [jnp.nan, jnp.nan],
            [1.1, -0.6],
            [jnp.nan, jnp.nan],
        ]
    )

    _, _, log_emit_seq = hmm_log_components(
        _build_hmm_dynamics(_delta_observation_model),
        obs_times,
        obs_values,
    )

    assert jnp.allclose(log_emit_seq[1], jnp.zeros(2))
    assert jnp.allclose(log_emit_seq[3], jnp.zeros(2))


def test_hmm_partial_missing_non_marginalizable_observation_family_raises():
    obs_times = jnp.arange(3.0)
    obs_values = jnp.array(
        [
            [-0.9, 0.4],
            [1.1, -0.6],
            [-0.2, 0.0],
        ]
    )
    obs_values = set_partial_row_missing(obs_values, 1, dim_idx=1)

    with pytest.raises(
        NotImplementedError,
        match="Partial missingness currently requires",
    ):
        hmm_log_components(
            _build_hmm_dynamics(_delta_observation_model),
            obs_times,
            obs_values,
        )


def test_hmm_partial_missing_independent_categorical_matches_manual_reference():
    obs_times = jnp.arange(2.0)
    obs_values = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    obs_values = set_partial_row_missing(obs_values, 0, dim_idx=1)

    _, _, log_emit_seq = hmm_log_components(
        _build_hmm_dynamics(_independent_categorical_observation_model),
        obs_times,
        obs_values,
        _obs_values_filled=jnp.array([[1, -1], [0, 1]], dtype=jnp.int32),
        _obs_mask=jnp.array([[True, False], [True, True]]),
    )

    probs = jnp.array(
        [
            [[0.85, 0.15], [0.7, 0.3]],
            [[0.2, 0.8], [0.35, 0.65]],
        ]
    )
    expected = jnp.array([jnp.log(probs[0, 0, 1]), jnp.log(probs[1, 0, 1])])

    assert jnp.allclose(log_emit_seq[0], expected)


def test_plated_hmm_gaussian_missingness_matches_memberwise_filter():
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

    tr = _run_plated_hmm_filter_trace(
        lambda **kwargs: _plated_hmm_model(_mvn_observation_model, **kwargs),
        obs_times=obs_times,
        obs_values=obs_values,
        M=M,
    )

    actual_log_filtered = tr["f_log_filtered_states"]["value"]
    actual_filtered = tr["f_filtered_states"]["value"]

    dynamics = _build_hmm_dynamics(_mvn_observation_model)
    expected_log_filtered = []
    for member_idx in range(M):
        loglik, log_filtered = compute_hmm_filter(
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values[member_idx],
        )
        assert jnp.isfinite(loglik)
        expected_log_filtered.append(log_filtered)

    expected_log_filtered = jnp.stack(expected_log_filtered, axis=0)
    assert actual_log_filtered.shape == (M, len(obs_times), 2)
    assert actual_filtered.shape == (M, len(obs_times), 2)
    assert jnp.allclose(actual_log_filtered, expected_log_filtered)
    assert jnp.allclose(actual_filtered, jnp.exp(expected_log_filtered))


def test_plated_hmm_independent_categorical_missingness_matches_memberwise_filter():
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

    tr = _run_plated_hmm_filter_trace(
        lambda **kwargs: _plated_hmm_model(
            _independent_categorical_observation_model,
            **kwargs,
        ),
        obs_times=obs_times,
        obs_values=obs_values,
        M=M,
    )

    actual_log_filtered = tr["f_log_filtered_states"]["value"]
    actual_filtered = tr["f_filtered_states"]["value"]

    dynamics = _build_hmm_dynamics(_independent_categorical_observation_model)
    expected_log_filtered = []
    for member_idx in range(M):
        loglik, log_filtered = compute_hmm_filter(
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values[member_idx],
        )
        assert jnp.isfinite(loglik)
        expected_log_filtered.append(log_filtered)

    expected_log_filtered = jnp.stack(expected_log_filtered, axis=0)
    assert actual_log_filtered.shape == (M, len(obs_times), 2)
    assert actual_filtered.shape == (M, len(obs_times), 2)
    assert jnp.allclose(actual_log_filtered, expected_log_filtered)
    assert jnp.allclose(actual_filtered, jnp.exp(expected_log_filtered))


def test_hmm_partial_missing_independent_categorical_keeps_filter_finite():
    obs_times = jnp.arange(4.0)
    obs_values = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    obs_values = set_partial_row_missing(obs_values, 1, dim_idx=0)

    tr = _run_hmm_filter_trace(
        _build_hmm_dynamics(_independent_categorical_observation_model),
        obs_times,
        obs_values,
    )

    filtered = tr["f_filtered_states"]["value"]

    assert filtered.shape == (len(obs_times), 2)
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
    assert jnp.allclose(filtered.sum(axis=-1), 1.0)


def test_hmm_full_row_missing_scalar_categorical_keeps_filter_finite():
    obs_times = jnp.arange(5.0)
    obs_values = jnp.array([0.0, 1.0, 3.0, 2.0, 1.0])
    obs_values = obs_values.at[1].set(jnp.nan)
    obs_values = obs_values.at[3].set(jnp.nan)

    tr = _run_hmm_filter_trace(
        _build_hmm_dynamics(_joint_categorical_observation_model),
        obs_times,
        obs_values,
    )

    filtered = tr["f_filtered_states"]["value"]

    assert filtered.shape == (len(obs_times), 2)
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
    assert jnp.allclose(filtered.sum(axis=-1), 1.0)


def test_hmm_fully_observed_scalar_categorical_integer_labels_match_float_path():
    obs_times = jnp.arange(5.0)
    obs_values_int = jnp.array([0, 1, 3, 2, 1], dtype=jnp.int32)
    obs_values_float = obs_values_int.astype(jnp.float32)

    tr_int = _run_hmm_filter_trace(
        _build_hmm_dynamics(_joint_categorical_observation_model),
        obs_times,
        obs_values_int,
    )
    tr_float = _run_hmm_filter_trace(
        _build_hmm_dynamics(_joint_categorical_observation_model),
        obs_times,
        obs_values_float,
    )

    assert jnp.isfinite(tr_int["f_marginal_loglik"]["value"])
    assert jnp.allclose(
        tr_int["f_marginal_loglik"]["value"],
        tr_float["f_marginal_loglik"]["value"],
    )
    assert jnp.allclose(
        tr_int["f_filtered_states"]["value"],
        tr_float["f_filtered_states"]["value"],
    )


def test_hmm_scalar_categorical_filter_mcmc_runs_under_tracing():
    obs_times = jnp.arange(20.0)
    true_A = jnp.array([[0.95, 0.05], [0.1, 0.9]])

    with DiscreteTimeSimulator():
        synthetic = Predictive(_scalar_categorical_hmm_model, num_samples=1)(
            jr.PRNGKey(0),
            A=true_A,
            predict_times=obs_times,
        )

    obs_values = jnp.asarray(synthetic["f_observations"])[0, 0, :, 0]

    with Filter(filter_config=HMMConfig()):
        mcmc = MCMC(
            NUTS(_scalar_categorical_hmm_model),
            num_warmup=1,
            num_samples=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(1), obs_times=obs_times, obs_values=obs_values)

    posterior = mcmc.get_samples()
    assert posterior["A"].shape == (1, 2, 2)


def test_hmm_independent_categorical_missing_filter_mcmc_runs_under_tracing():
    obs_times = jnp.arange(10.0)
    true_A = jnp.array([[0.92, 0.08], [0.12, 0.88]])

    def model(A=None, obs_times=None, obs_values=None, predict_times=None):
        A = numpyro.sample(
            "A",
            dist.Dirichlet(jnp.ones(2)).expand([2]).to_event(1),
            obs=A,
        )

        dynamics = DynamicalModel(
            control_dim=0,
            initial_condition=dist.Categorical(probs=jnp.array([0.5, 0.5])),
            state_evolution=lambda x, u, t_now, t_next: dist.Categorical(probs=A[x]),
            observation_model=_independent_categorical_observation_model,
        )

        return dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )

    with DiscreteTimeSimulator():
        synthetic = Predictive(
            model,
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(2), A=true_A, predict_times=obs_times)

    obs_values = jnp.asarray(synthetic["f_observations"])[0, 0].astype(float)
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_full_row_missing(obs_values, 3)
    obs_values = set_partial_row_missing(obs_values, 6, dim_idx=1)

    with Filter(filter_config=HMMConfig(record_filtered=True)):
        mcmc = MCMC(
            NUTS(model),
            num_warmup=1,
            num_samples=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(3), obs_times=obs_times, obs_values=obs_values)
        posterior = mcmc.get_samples()

        predictive = Predictive(
            model,
            params={"A": posterior["A"].mean(axis=0)},
            num_samples=1,
            exclude_deterministic=False,
        )
        filtered = predictive(
            jr.PRNGKey(4),
            obs_times=obs_times,
            obs_values=obs_values,
        )

    # This covers the traced NUTS/Predictive path that previously hit
    # NonConcreteBooleanIndexError for categorical observation masks.
    assert posterior["A"].shape == (1, 2, 2)
    assert jnp.asarray(filtered["f_filtered_states"]).shape[-2:] == (len(obs_times), 2)


def test_hmm_categorical_observations_reject_negative_labels_early():
    obs_times = jnp.arange(3.0)
    obs_values = jnp.array([0.0, -1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="zero-based integer labels 0..K-1; found negative observed value",
    ):
        _run_hmm_filter_trace(
            _build_hmm_dynamics(_joint_categorical_observation_model),
            obs_times,
            obs_values,
        )


def test_hmm_categorical_observations_reject_non_integer_labels_early():
    obs_times = jnp.arange(3.0)
    obs_values = jnp.array([0.0, 1.5, 2.0])

    with pytest.raises(
        ValueError,
        match="zero-based integer labels. Found non-integer observed value",
    ):
        _run_hmm_filter_trace(
            _build_hmm_dynamics(_joint_categorical_observation_model),
            obs_times,
            obs_values,
        )


def test_hmm_categorical_observations_reject_out_of_range_labels_early():
    obs_times = jnp.arange(3.0)
    obs_values = jnp.array([0.0, 4.0, 2.0])

    with pytest.raises(
        ValueError,
        match="zero-based integer labels 0..K-1.*with K=4",
    ):
        _run_hmm_filter_trace(
            _build_hmm_dynamics(_joint_categorical_observation_model),
            obs_times,
            obs_values,
        )
