import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from jax.experimental import sparse as jax_sparse
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator
from dynestyx.inference.configs.filter import (
    ContinuousTimeDPFConfig,
    EKFConfig,
    EnKFConfig,
    KFConfig,
    PFConfig,
)
from dynestyx.inference.filters import Filter
from dynestyx.inference.integrations.cuthbert.discrete import (
    compute_cuthbert_filter,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    run_discrete_filter as run_cuthbert_discrete_filter,
)
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    LinearGaussianStateEvolution,
)
from dynestyx.models.observations import LinearGaussianObservation
from tests.fixtures import (
    _squeeze_sim_dims,
    data_conditioned_jumpy_controls,
    data_conditioned_jumpy_controls_ode,
    data_conditioned_jumpy_controls_sde,
)
from tests.models import discrete_time_l63_model, discrete_time_lti_simplified_model
from tests.test_utils import (
    assert_trace_sites_exist_and_field_all_finite,
    assert_tree_all_finite,
)


@pytest.mark.parametrize(
    ("filter_type", "filter_source", "mean_error_tol"),
    [
        ("kf", "cuthbert", 1e-1),
        ("kf", "cd_dynamax", 1e-1),
        ("ekf", "cuthbert", 1e-1),
        ("enkf", "cuthbert", 2e-1),
        ("ekf", "cd_dynamax", 1e-1),
        ("ukf", "cd_dynamax", 1e-1),
        ("pf", "cuthbert", 1e-1),
    ],
)
def test_jumpy_controls(filter_type, filter_source, mean_error_tol):
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls(
        filter_type=filter_type,
        filter_source=filter_source,
    )
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic[
        "observations"
    ]  # (T, obs_dim) after _normalize_synthetic
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < mean_error_tol


def test_jumpy_controls_sde():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_sde()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic[
        "observations"
    ]  # (T, obs_dim) after _normalize_synthetic
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 3.5e-2


def test_jumpy_controls_ode():
    data_conditioned_model, synthetic = data_conditioned_jumpy_controls_ode()
    rng_key = jr.PRNGKey(0)
    with trace() as tr, seed(rng_seed=rng_key):
        data_conditioned_model()

    synthetic_observations = synthetic[
        "observations"
    ]  # (T, obs_dim) after _normalize_synthetic
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert synthetic_observations.shape == filtered_means.shape
    assert jnp.allclose(synthetic_observations, filtered_means, atol=1e0)
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 0.1


def test_continuous_time_dpf_non_gaussian_observation_smoke():
    obs_times = jnp.array([0.0, 0.1, 0.2], dtype=jnp.float32)
    obs_values = jnp.array([0, 1, 0], dtype=jnp.int32)

    def model():
        bias = numpyro.sample("bias", dist.Normal(0.0, 0.5))
        dynamics = DynamicalModel(
            initial_condition=dist.LogNormal(loc=jnp.zeros(1), scale=jnp.ones(1)),
            state_evolution=ContinuousTimeStateEvolution(
                drift=lambda x, u, t: -0.3 * jnp.sin(x),
                diffusion=FullDiffusion(lambda x, u, t: 0.1 * jnp.eye(1)),
            ),
            observation_model=lambda x, u, t: dist.Poisson(rate=jnp.exp(x[0] + bias)),
        )
        dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    with seed(rng_seed=jr.PRNGKey(0)):
        with Filter(filter_config=ContinuousTimeDPFConfig(n_particles=32)):
            model()


def _make_discrete_lti_data():
    obs_times = jnp.arange(start=0.0, stop=6.0, step=1.0)
    true_params = {"alpha": jnp.array(0.35)}
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(jr.PRNGKey(0), predict_times=obs_times)
    return obs_times, _squeeze_sim_dims(synthetic["f_observations"])


def _make_discrete_lti_dynamics(alpha=0.35):
    state_dim = 2
    return dsx.LTI_discrete(
        A=jnp.array([[alpha, 0.1], [0.1, 0.8]]),
        Q=0.1 * jnp.eye(state_dim),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.5**2]]),
        B=jnp.array([[0.1], [0.0]]),
        D=jnp.array([[0.01]]),
    )


def _covariance_from_cholesky(chol_cov):
    return chol_cov @ jnp.swapaxes(chol_cov, -1, -2)


@pytest.mark.parametrize(
    "filter_config",
    [
        KFConfig(filter_source="cuthbert"),
        KFConfig(filter_source="cuthbert", associative=True),
        EKFConfig(filter_source="cuthbert"),
        EnKFConfig(n_particles=16, filter_source="cuthbert"),
        PFConfig(n_particles=16, filter_source="cuthbert"),
    ],
)
def test_compute_cuthbert_filter_returns_observation_aligned_states(filter_config):
    obs_times, obs_values = _make_discrete_lti_data()
    dynamics = _make_discrete_lti_dynamics()

    marginal_loglik, states = compute_cuthbert_filter(
        dynamics,
        filter_config,
        key=jr.PRNGKey(2),
        obs_times=obs_times,
        obs_values=obs_values,
    )

    assert jnp.ndim(marginal_loglik) == 0
    assert_tree_all_finite({"marginal_loglik": marginal_loglik}, where="filter output")
    assert states.log_normalizing_constant.shape[0] == len(obs_times)
    assert states.model_inputs.y.shape[0] == len(obs_times)

    if isinstance(filter_config, PFConfig):
        assert_tree_all_finite(
            {
                "particles": states.particles,
                "log_weights": states.log_weights,
                "log_normalizing_constant": states.log_normalizing_constant,
            },
            where="PF filter states",
        )
        assert states.particles.shape[0] == len(obs_times)
        assert states.log_weights.shape[0] == len(obs_times)
    else:
        assert_tree_all_finite(
            {
                "mean": states.mean,
                "chol_cov": states.chol_cov,
                "log_normalizing_constant": states.log_normalizing_constant,
            },
            where="Gaussian filter states",
        )
        assert states.mean.shape[0] == len(obs_times)
        assert states.chol_cov.shape[0] == len(obs_times)


def test_compute_cuthbert_filter_can_return_raw_cuthbert_states():
    obs_times, obs_values = _make_discrete_lti_data()
    dynamics = _make_discrete_lti_dynamics()

    _, states = compute_cuthbert_filter(
        dynamics,
        KFConfig(filter_source="cuthbert"),
        obs_times=obs_times,
        obs_values=obs_values,
        align_to_observations=False,
    )

    assert states.log_normalizing_constant.shape[0] == len(obs_times) + 1
    assert states.model_inputs.y.shape[0] == len(obs_times) + 1
    assert states.mean.shape[0] == len(obs_times) + 1
    assert states.chol_cov.shape[0] == len(obs_times) + 1
    assert_tree_all_finite(
        {
            "log_normalizing_constant": states.log_normalizing_constant,
            "mean": states.mean,
            "chol_cov": states.chol_cov,
        },
        where="raw cuthbert filter states",
    )


def test_cuthbert_enkf_prediction_storage_follows_collection_and_explicit_override():
    obs_times, obs_values = _make_discrete_lti_data()
    dynamics = _make_discrete_lti_dynamics()
    key = jr.PRNGKey(17)

    def run_filter(
        filter_config: EnKFConfig,
        *,
        store_predicted_ensemble: bool | None = None,
    ):
        return compute_cuthbert_filter(
            dynamics,
            filter_config,
            key,
            obs_times=obs_times,
            obs_values=obs_values,
            store_predicted_ensemble=store_predicted_ensemble,
        )

    loglik_disabled, states_disabled = run_filter(
        EnKFConfig(
            n_particles=16,
            include_predicted_observations=False,
        ),
    )
    loglik_enabled, states_enabled = run_filter(
        EnKFConfig(
            n_particles=16,
            include_predicted_observations=True,
        ),
    )
    loglik_forced_disabled, states_forced_disabled = run_filter(
        EnKFConfig(
            n_particles=16,
            include_predicted_observations=True,
        ),
        store_predicted_ensemble=False,
    )
    loglik_forced_enabled, states_forced_enabled = run_filter(
        EnKFConfig(
            n_particles=16,
            include_predicted_observations=False,
        ),
        store_predicted_ensemble=True,
    )

    assert states_disabled.predicted_ensemble is None
    assert states_forced_disabled.predicted_ensemble is None
    assert states_enabled.predicted_ensemble.shape == (
        len(obs_times),
        16,
        dynamics.state_dim,
    )
    assert states_forced_enabled.predicted_ensemble.shape == (
        len(obs_times),
        16,
        dynamics.state_dim,
    )

    for loglik in (
        loglik_enabled,
        loglik_forced_disabled,
        loglik_forced_enabled,
    ):
        assert jnp.array_equal(loglik, loglik_disabled)
    for states in (
        states_enabled,
        states_forced_disabled,
        states_forced_enabled,
    ):
        assert jnp.array_equal(states.ensemble, states_disabled.ensemble)


def test_cuthbert_enkf_predicted_ensemble_drops_only_the_leading_dummy_step():
    obs_times, obs_values = _make_discrete_lti_data()
    dynamics = _make_discrete_lti_dynamics()
    filter_config = EnKFConfig(
        n_particles=16,
        include_predicted_observations=False,
    )

    def run_filter(*, align_to_observations: bool = True):
        return compute_cuthbert_filter(
            dynamics,
            filter_config,
            jr.PRNGKey(23),
            obs_times=obs_times,
            obs_values=obs_values,
            store_predicted_ensemble=True,
            align_to_observations=align_to_observations,
        )

    _, aligned = run_filter()
    _, raw = run_filter(align_to_observations=False)

    assert aligned.predicted_ensemble.shape == (
        len(obs_times),
        filter_config.n_particles,
        dynamics.state_dim,
    )
    assert raw.predicted_ensemble.shape == (
        len(obs_times) + 1,
        filter_config.n_particles,
        dynamics.state_dim,
    )
    assert jnp.array_equal(
        aligned.predicted_ensemble,
        raw.predicted_ensemble[1:],
    )
    assert jnp.array_equal(aligned.model_inputs.time, obs_times)
    assert jnp.array_equal(aligned.model_inputs.y, obs_values)
    assert not jnp.allclose(
        aligned.predicted_ensemble[0],
        aligned.ensemble[0],
    )


@pytest.mark.parametrize(
    "filter_config",
    [
        KFConfig(filter_source="cuthbert"),
        KFConfig(filter_source="cuthbert", associative=True),
        EKFConfig(filter_source="cuthbert"),
        EnKFConfig(n_particles=16, filter_source="cuthbert"),
        PFConfig(n_particles=16, filter_source="cuthbert"),
    ],
)
def test_cuthbert_filtered_distribution_shapes_match_observations(filter_config):
    obs_times, obs_values = _make_discrete_lti_data()
    dynamics = _make_discrete_lti_dynamics()

    _marginal_loglik, _states, filtered_dists = run_cuthbert_discrete_filter(
        "f",
        dynamics,
        filter_config,
        key=jr.PRNGKey(2),
        obs_times=obs_times,
        obs_values=obs_values,
    )

    assert len(filtered_dists) == len(obs_times)
    for filtered_dist in filtered_dists:
        assert filtered_dist.event_shape == (dynamics.state_dim,)

        if isinstance(filter_config, PFConfig):
            assert_tree_all_finite(
                {
                    "particles": filtered_dist.particles,
                    "log_weights": filtered_dist.log_weights,
                },
                where="filtered distribution",
            )
            assert filtered_dist.particles.shape == (
                filter_config.n_particles,
                dynamics.state_dim,
            )
            assert filtered_dist.log_weights.shape == (filter_config.n_particles,)
        else:
            assert_tree_all_finite(
                {
                    "mean": filtered_dist.mean,
                    "covariance_matrix": filtered_dist.covariance_matrix,
                },
                where="filtered distribution",
            )
            assert filtered_dist.mean.shape == (dynamics.state_dim,)
            assert filtered_dist.covariance_matrix.shape == (
                dynamics.state_dim,
                dynamics.state_dim,
            )


def test_cuthbert_associative_kf_matches_sequential():
    obs_times, obs_values = _make_discrete_lti_data()
    dynamics = _make_discrete_lti_dynamics()

    seq_marginal_loglik, seq_states = compute_cuthbert_filter(
        dynamics,
        KFConfig(filter_source="cuthbert"),
        obs_times=obs_times,
        obs_values=obs_values,
    )
    assoc_marginal_loglik, assoc_states = compute_cuthbert_filter(
        dynamics,
        KFConfig(filter_source="cuthbert", associative=True),
        obs_times=obs_times,
        obs_values=obs_values,
    )

    assert jnp.allclose(
        seq_marginal_loglik, assoc_marginal_loglik, rtol=1e-6, atol=1e-6
    )
    assert jnp.allclose(seq_states.mean, assoc_states.mean, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(
        _covariance_from_cholesky(seq_states.chol_cov),
        _covariance_from_cholesky(assoc_states.chol_cov),
        rtol=1e-6,
        atol=1e-6,
    )
    assert jnp.allclose(
        seq_states.log_normalizing_constant,
        assoc_states.log_normalizing_constant,
        rtol=1e-6,
        atol=1e-6,
    )


def test_kf_config_rejects_associative_outside_cuthbert():
    with pytest.raises(ValueError, match="filter_source='cuthbert'"):
        KFConfig(filter_source="cd_dynamax", associative=True)


def test_cuthbert_enkf_accepts_callable_independent_normal_observation():
    obs_times = jnp.arange(start=0.0, stop=4.0, step=1.0)
    obs_values = jnp.zeros((len(obs_times), 2))
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=jnp.eye(2)
        ),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            loc=0.9 * x,
            covariance_matrix=0.1 * jnp.eye(2),
        ),
        observation_model=lambda x, u, t: dist.Independent(
            dist.Normal(loc=x, scale=0.2), 1
        ),
    )

    _marginal_loglik, _states, filtered_dists = run_cuthbert_discrete_filter(
        "f",
        dynamics,
        EnKFConfig(n_particles=16, filter_source="cuthbert"),
        key=jr.PRNGKey(2),
        obs_times=obs_times,
        obs_values=obs_values,
    )

    assert len(filtered_dists) == len(obs_times)
    assert all(d.event_shape == (dynamics.state_dim,) for d in filtered_dists)


def test_cuthbert_enkf_rejects_state_dependent_observation_noise():
    obs_times = jnp.arange(start=0.0, stop=4.0, step=1.0)
    obs_values = jnp.zeros((len(obs_times), 2))
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=jnp.eye(2)
        ),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            loc=0.9 * x, covariance_matrix=0.1 * jnp.eye(2)
        ),
        observation_model=lambda x, u, t: dist.Independent(
            dist.Normal(loc=x, scale=0.1 + 0.5 * jnp.abs(x)), 1
        ),
    )

    with pytest.raises(ValueError, match="state-independent"):
        with trace(), seed(rng_seed=jr.PRNGKey(1)):
            run_cuthbert_discrete_filter(
                "f",
                dynamics,
                EnKFConfig(n_particles=16, filter_source="cuthbert"),
                key=jr.PRNGKey(2),
                obs_times=obs_times,
                obs_values=obs_values,
            )


def test_cuthbert_enkf_records_filtered_gaussian_sites():
    obs_times, obs_values = _make_discrete_lti_data()
    substituted = numpyro.handlers.substitute(
        discrete_time_lti_simplified_model, data={"alpha": jnp.array(0.35)}
    )
    filter_config = EnKFConfig(
        n_particles=16,
        filter_source="cuthbert",
        record_filtered_states_mean=True,
        record_filtered_states_cov=True,
        record_filtered_states_cov_diag=True,
        record_filtered_states_chol_cov=True,
    )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(1)):
        with Filter(filter_config=filter_config):
            substituted(obs_times=obs_times, obs_values=obs_values)

    assert_trace_sites_exist_and_field_all_finite(
        tr,
        "f_marginal_loglik",
        "f_filtered_states_mean",
        "f_filtered_states_cov",
        "f_filtered_states_cov_diag",
        "f_filtered_states_chol_cov",
        where="recorded filter trace",
    )
    assert tr["f_filtered_states_mean"]["value"].shape == (len(obs_times), 2)
    assert tr["f_filtered_states_cov"]["value"].shape == (len(obs_times), 2, 2)
    assert tr["f_filtered_states_cov_diag"]["value"].shape == (len(obs_times), 2)
    assert tr["f_filtered_states_chol_cov"]["value"].shape[:2] == (
        len(obs_times),
        2,
    )


def test_cuthbert_enkf_fixed_crn_seed_is_deterministic():
    obs_times, obs_values = _make_discrete_lti_data()
    substituted = numpyro.handlers.substitute(
        discrete_time_lti_simplified_model, data={"alpha": jnp.array(0.35)}
    )
    filter_config = EnKFConfig(
        n_particles=16, filter_source="cuthbert", crn_seed=jr.PRNGKey(123)
    )

    def _run(seed_key):
        with trace() as tr, seed(rng_seed=seed_key):
            with Filter(filter_config=filter_config):
                substituted(obs_times=obs_times, obs_values=obs_values)
        return tr["f_marginal_loglik"]["value"]

    assert jnp.isclose(_run(jr.PRNGKey(1)), _run(jr.PRNGKey(2)))


def test_cuthbert_enkf_crn_seed_none_uses_numpyro_seed():
    obs_times, obs_values = _make_discrete_lti_data()
    substituted = numpyro.handlers.substitute(
        discrete_time_lti_simplified_model, data={"alpha": jnp.array(0.35)}
    )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(1)):
        with Filter(filter_config=EnKFConfig(n_particles=16, crn_seed=None)):
            substituted(obs_times=obs_times, obs_values=obs_values)

    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])


def test_default_discrete_filter_uses_cuthbert_enkf_on_nonlinear_model():
    obs_times = jnp.arange(start=0.0, stop=1.0, step=0.2)
    true_params = {"rho": jnp.array(28.0)}
    predictive = Predictive(
        discrete_time_l63_model,
        params=true_params,
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        synthetic = predictive(jr.PRNGKey(0), predict_times=obs_times)
    obs_values = _squeeze_sim_dims(synthetic["f_observations"])
    substituted = numpyro.handlers.substitute(
        discrete_time_l63_model, data={"rho": jnp.array(28.0)}
    )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(2)):
        with Filter():
            substituted(obs_times=obs_times, obs_values=obs_values)

    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])


# --- sparse H (LinearGaussianObservation) ----------------------------------


def _sparse_h_test_dynamics(H) -> DynamicalModel:
    """A small 3-state, LTI model observing 2 (of 3) state components via H.

    Built directly from `LinearGaussianStateEvolution`/`LinearGaussianObservation` rather
    than the `LTI_discrete` factory, since `LTI_discrete`'s own `H` parameter is typed as a
    dense `Float[Array, ...]` and isn't part of this change's scope.
    """
    A = jnp.array([[0.9, 0.05, 0.0], [0.0, 0.85, 0.05], [0.0, 0.0, 0.8]])
    Q = 0.05 * jnp.eye(3)
    R = 0.1**2 * jnp.eye(H.shape[0])
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(3), jnp.eye(3)),
        state_evolution=LinearGaussianStateEvolution(A=A, cov=Q),
        observation_model=LinearGaussianObservation(H=H, R=R),
        control_dim=0,
    )


def test_linear_gaussian_observation_sparse_h_matches_dense_h():
    """A sparse H must give the same observation mean as its dense equivalent --
    exercises the `H @ x` fix in LinearGaussianObservation.__call__ directly."""
    H_dense = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    H_sparse = jax_sparse.BCOO.fromdense(H_dense)
    x = jnp.array([0.4, -0.2, 1.1])

    obs_dense = LinearGaussianObservation(H=H_dense, R=0.01 * jnp.eye(2))(x, None, 0.0)
    obs_sparse = LinearGaussianObservation(H=H_sparse, R=0.01 * jnp.eye(2))(
        x, None, 0.0
    )

    assert jnp.allclose(obs_dense.mean, obs_sparse.mean)


@pytest.mark.xfail(
    raises=ValueError,
    strict=True,
    reason="Sparse observation matrices are not supported inside dsx.plate.",
)
def test_sparse_h_in_plate():
    dynamics = _sparse_h_test_dynamics(jax_sparse.BCOO.fromdense(jnp.eye(2, 3)))
    with dsx.plate("members", 1):
        dsx.sample("f", dynamics, predict_times=jnp.arange(1.0))


def test_cuthbert_enkf_sparse_h_matches_dense_h():
    """EnKF with a sparse H must give the same marginal_loglik and filtered means as the
    dense H -- exercises the `_as_array_or_sparse` fix in the cuthbert EnKF backend."""
    H_dense = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    H_sparse = jax_sparse.BCOO.fromdense(H_dense)

    dynamics_dense = _sparse_h_test_dynamics(H_dense)
    dynamics_sparse = _sparse_h_test_dynamics(H_sparse)

    obs_times = jnp.arange(6.0)
    ground_truth = dsx.simulate(
        dynamics_dense, rng_key=jr.PRNGKey(0), predict_times=obs_times, n_simulations=1
    )
    obs_values = jnp.asarray(ground_truth.observations)[0]

    # Fixed crn_seed: EnKF's randomness becomes a deterministic function of its inputs, so
    # dense and sparse H should give numerically identical results, not just close ones.
    crn_seed = jr.PRNGKey(42)
    with Filter(filter_config=EnKFConfig(n_particles=32, crn_seed=crn_seed)):
        result_dense = dsx.condition(
            "f", dynamics_dense, obs_times=obs_times, obs_values=obs_values
        )
    with Filter(filter_config=EnKFConfig(n_particles=32, crn_seed=crn_seed)):
        result_sparse = dsx.condition(
            "f", dynamics_sparse, obs_times=obs_times, obs_values=obs_values
        )

    assert jnp.allclose(result_dense.marginal_loglik, result_sparse.marginal_loglik)
    means_dense = jnp.stack([d.mean for d in result_dense.dists])
    means_sparse = jnp.stack([d.mean for d in result_sparse.dists])
    assert jnp.allclose(means_dense, means_sparse)


def test_cuthbert_enkf_filtered_dists_are_low_rank_and_samplable():
    """EnKF filtered distributions keep the ensemble factor and stay samplable.

    The ensemble covariance has rank at most `n_particles - 1`, so expanding it into
    a dense `MultivariateNormal` gives a singular matrix whose eager Cholesky is
    `nan` -- which silently propagated into posterior rollout, since the rollout
    grafts these distributions in as the forecast initial condition and samples
    them. `n_particles=3` against a 3-state model puts us in that rank-deficient
    regime on purpose.
    """
    dynamics = _sparse_h_test_dynamics(jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    obs_times = jnp.arange(6.0)
    ground_truth = dsx.simulate(
        dynamics, rng_key=jr.PRNGKey(0), predict_times=obs_times, n_simulations=1
    )
    obs_values = jnp.asarray(ground_truth.observations)[0]

    n_particles = 3
    with Filter(
        filter_config=EnKFConfig(n_particles=n_particles, crn_seed=jr.PRNGKey(42))
    ):
        result = dsx.condition(
            "f", dynamics, obs_times=obs_times, obs_values=obs_values
        )

    ensemble = result.states.ensemble
    assert len(result.dists) == len(obs_times)
    for t, d in enumerate(result.dists):
        assert isinstance(d, dist.LowRankMultivariateNormal)
        assert d.event_shape == (dynamics.state_dim,)
        assert d.cov_factor.shape == (dynamics.state_dim, n_particles)
        assert jnp.allclose(d.mean, ensemble[t].mean(axis=0), atol=1e-5)
        assert jnp.isfinite(d.sample(jr.PRNGKey(t))).all()


def test_kf_raises_on_sparse_observation_matrix_cuthbert():
    """KF's cuthbert backend does not support a sparse H: cuthbert's Kalman internals
    cannot handle it (vmap's sparse tracing breaks inside a jnp.block call), so dynestyx
    raises a clear error up front rather than letting that confusing internal failure
    surface."""
    H_sparse = jax_sparse.BCOO.fromdense(jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    dynamics_sparse = _sparse_h_test_dynamics(H_sparse)
    obs_times = jnp.arange(3.0)
    obs_values = jnp.zeros((3, 2))

    with pytest.raises(ValueError, match="not supported"):
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
            dsx.condition(
                "f", dynamics_sparse, obs_times=obs_times, obs_values=obs_values
            )


def test_compute_cuthbert_filter_raises_on_sparse_observation_matrix_direct_call():
    """The sparse-H check must live at the actual cuthbert entry point
    (_cuthbert_filter_kalman), not only in Filter: compute_cuthbert_filter is called
    directly elsewhere (this test file, test_time_varying_linear_gaussian.py, and the
    cuthbert smoother), bypassing Filter entirely. A direct call must raise the same
    clear error rather than surfacing a confusing internal AssertionError."""
    H_sparse = jax_sparse.BCOO.fromdense(jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    dynamics_sparse = _sparse_h_test_dynamics(H_sparse)
    obs_times = jnp.arange(3.0)
    obs_values = jnp.zeros((3, 2))

    with pytest.raises(ValueError, match="not supported"):
        compute_cuthbert_filter(
            dynamics_sparse,
            KFConfig(filter_source="cuthbert"),
            key=jr.PRNGKey(0),
            obs_times=obs_times,
            obs_values=obs_values,
        )


def test_kf_sparse_h_matches_dense_h_cd_dynamax():
    """Unlike the cuthbert backend, KF's cd_dynamax backend handles a sparse H fine:
    dynamax's own Kalman update only ever uses H in plain matmuls (H @ P @ H.T, etc.),
    never mixed into a jnp.block/concatenate call, so there's no sparse-tracing collision.
    marginal_loglik should match the dense H case exactly (deterministic filter, no CRN
    needed)."""
    H_dense = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    H_sparse = jax_sparse.BCOO.fromdense(H_dense)
    dynamics_dense = _sparse_h_test_dynamics(H_dense)
    dynamics_sparse = _sparse_h_test_dynamics(H_sparse)
    obs_times = jnp.arange(6.0)
    ground_truth = dsx.simulate(
        dynamics_dense, rng_key=jr.PRNGKey(0), predict_times=obs_times, n_simulations=1
    )
    obs_values = jnp.asarray(ground_truth.observations)[0]

    with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
        result_dense = dsx.condition(
            "f", dynamics_dense, obs_times=obs_times, obs_values=obs_values
        )
    with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
        result_sparse = dsx.condition(
            "f", dynamics_sparse, obs_times=obs_times, obs_values=obs_values
        )
    assert jnp.allclose(result_dense.marginal_loglik, result_sparse.marginal_loglik)


def test_ekf_warns_on_sparse_observation_matrix_but_still_works():
    """EKF works correctly with a sparse H, but the warning should fire to flag that
    there's likely no efficiency gain (EKF differentiates the log-density directly rather
    than extracting H, so the underlying computation is dense-scaling regardless)."""
    H_dense = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    H_sparse = jax_sparse.BCOO.fromdense(H_dense)
    dynamics_dense = _sparse_h_test_dynamics(H_dense)
    dynamics_sparse = _sparse_h_test_dynamics(H_sparse)
    obs_times = jnp.arange(6.0)
    ground_truth = dsx.simulate(
        dynamics_dense, rng_key=jr.PRNGKey(0), predict_times=obs_times, n_simulations=1
    )
    obs_values = jnp.asarray(ground_truth.observations)[0]

    with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
        result_dense = dsx.condition(
            "f", dynamics_dense, obs_times=obs_times, obs_values=obs_values
        )
    with pytest.warns(UserWarning, match="no efficiency gain"):
        with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
            result_sparse = dsx.condition(
                "f", dynamics_sparse, obs_times=obs_times, obs_values=obs_values
            )
    assert jnp.allclose(result_dense.marginal_loglik, result_sparse.marginal_loglik)


def test_ekf_warns_on_sparse_observation_matrix_but_still_works_cd_dynamax():
    """EKF's cd_dynamax backend also works correctly with a sparse H, and also warns of
    likely no efficiency gain: cd_dynamax's EKF computes H as
    jax.jacfwd(emission_function), which materializes a dense Jacobian at every step
    regardless of H's sparsity (same root cause as the cuthbert EKF warning, different
    mechanism)."""
    H_dense = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    H_sparse = jax_sparse.BCOO.fromdense(H_dense)
    dynamics_dense = _sparse_h_test_dynamics(H_dense)
    dynamics_sparse = _sparse_h_test_dynamics(H_sparse)
    obs_times = jnp.arange(6.0)
    ground_truth = dsx.simulate(
        dynamics_dense, rng_key=jr.PRNGKey(0), predict_times=obs_times, n_simulations=1
    )
    obs_values = jnp.asarray(ground_truth.observations)[0]

    with Filter(filter_config=EKFConfig(filter_source="cd_dynamax")):
        result_dense = dsx.condition(
            "f", dynamics_dense, obs_times=obs_times, obs_values=obs_values
        )
    with pytest.warns(UserWarning, match="no efficiency gain"):
        with Filter(filter_config=EKFConfig(filter_source="cd_dynamax")):
            result_sparse = dsx.condition(
                "f", dynamics_sparse, obs_times=obs_times, obs_values=obs_values
            )
    assert jnp.allclose(result_dense.marginal_loglik, result_sparse.marginal_loglik)
