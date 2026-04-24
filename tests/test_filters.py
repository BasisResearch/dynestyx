import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx.inference.filter_configs import (
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
from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.simulators import DiscreteTimeSimulator
from tests.fixtures import (
    _squeeze_sim_dims,
    data_conditioned_jumpy_controls,
    data_conditioned_jumpy_controls_ode,
    data_conditioned_jumpy_controls_sde,
)
from tests.models import discrete_time_l63_model, discrete_time_lti_simplified_model


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
    assert jnp.abs(jnp.mean(synthetic_observations - filtered_means)) < 3e-2


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
                diffusion_coefficient=lambda x, u, t: 0.1 * jnp.eye(1),
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
    assert states.log_normalizing_constant.shape[0] == len(obs_times)
    assert states.model_inputs.y.shape[0] == len(obs_times)

    if isinstance(filter_config, PFConfig):
        assert states.particles.shape[0] == len(obs_times)
        assert states.log_weights.shape[0] == len(obs_times)
    else:
        assert states.mean.shape[0] == len(obs_times)
        assert states.chol_cov.shape[0] == len(obs_times)


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

    with trace(), seed(rng_seed=jr.PRNGKey(1)):
        filtered_dists = run_cuthbert_discrete_filter(
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
            assert filtered_dist.particles.shape == (
                filter_config.n_particles,
                dynamics.state_dim,
            )
            assert filtered_dist.log_weights.shape == (filter_config.n_particles,)
        else:
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

    with trace(), seed(rng_seed=jr.PRNGKey(1)):
        filtered_dists = run_cuthbert_discrete_filter(
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

    assert "f_marginal_loglik" in tr
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
