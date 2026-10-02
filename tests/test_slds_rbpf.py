import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, DynamicalModel, Filter
from dynestyx.inference.filters import RBPFConfig
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import NUTSConfig
from dynestyx.models import (
    MixedStateDistribution,
    SwitchingLinearGaussianObservation,
    SwitchingLinearGaussianStateEvolution,
)


def _small_slds_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    ctrl_times=None,
    ctrl_values=None,
    *,
    sample_bias_shift=False,
):
    num_regimes = 2
    state_dim = 2
    observation_dim = 1
    transition_matrix = jnp.array([[0.95, 0.05], [0.10, 0.90]])
    dynamics_matrices = jnp.array(
        [
            [[0.95, 0.10], [-0.05, 0.90]],
            [[0.55, -0.25], [0.20, 0.65]],
        ]
    )
    dynamics_covariances = jnp.tile(0.05**2 * jnp.eye(state_dim), (num_regimes, 1, 1))
    dynamics_biases = jnp.array([[0.0, 0.0], [0.75, -0.25]])
    if sample_bias_shift:
        bias_shift = numpyro.sample("bias_shift", dist.Normal(0.0, 0.1))
        dynamics_biases = dynamics_biases + bias_shift * jnp.array(
            [[0.0, 0.0], [1.0, 0.0]]
        )
    emission_matrices = jnp.tile(jnp.array([[1.0, 0.0]]), (num_regimes, 1, 1))
    emission_covariances = jnp.tile(
        0.10**2 * jnp.eye(observation_dim), (num_regimes, 1, 1)
    )
    emission_biases = jnp.zeros((num_regimes, observation_dim))
    initial_probs = jnp.array([0.80, 0.20])
    initial_mean = jnp.zeros((num_regimes, state_dim))
    initial_cov = jnp.tile(jnp.eye(state_dim), (num_regimes, 1, 1))

    state_evolution = SwitchingLinearGaussianStateEvolution(
        transition_matrix=transition_matrix,
        A=dynamics_matrices,
        cov=dynamics_covariances,
        bias=dynamics_biases,
    )
    observation_model = SwitchingLinearGaussianObservation(
        H=emission_matrices,
        R=emission_covariances,
        bias=emission_biases,
    )
    dynamics = DynamicalModel(
        initial_condition=MixedStateDistribution(
            categorical_probs=initial_probs,
            continuous_locs=initial_mean,
            continuous_covs=initial_cov,
        ),
        state_evolution=state_evolution,
        observation_model=observation_model,
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def _small_slds_model_with_parameter(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    ctrl_times=None,
    ctrl_values=None,
):
    return _small_slds_model(
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        sample_bias_shift=True,
    )


def _batched_small_slds_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    ctrl_times=None,
    ctrl_values=None,
):
    with dsx.plate("trajectories", 2):
        return _small_slds_model(
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


def _make_slds_observations():
    times = jnp.arange(8.0)
    with DiscreteTimeSimulator():
        prior = Predictive(
            _small_slds_model,
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(0), predict_times=times)
    return prior["f_times"][0, 0], prior["f_observations"][0, 0]


def test_mixed_state_distribution_delegates_log_prob_to_numpyro_distributions():
    probs = jnp.array([0.25, 0.75])
    locs = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    covs = jnp.stack([0.5 * jnp.eye(2), 1.5 * jnp.eye(2)])
    mixed = MixedStateDistribution(probs, locs, covs)

    value = jnp.array([1.0, 1.7, -0.4])
    expected = dist.Categorical(probs=probs).log_prob(1) + dist.MultivariateNormal(
        locs[1], covariance_matrix=covs[1]
    ).log_prob(value[1:])

    assert mixed.event_shape == (3,)
    assert jnp.allclose(mixed.log_prob(value), expected)
    assert mixed.sample(jr.PRNGKey(0), sample_shape=(5,)).shape == (5, 3)


def test_slds_rbpf_non_batched_predictive_trace_sites_are_finite():
    obs_times, obs_values = _make_slds_observations()
    with Filter(
        RBPFConfig(
            n_particles=64,
            proposal="optimal",
            record_filtered_states_mean=True,
            record_filtered_regime_probs=True,
            crn_seed=jr.PRNGKey(1),
        )
    ):
        out = Predictive(
            _small_slds_model,
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(2), obs_times=obs_times, obs_values=obs_values)

    assert jnp.isfinite(out["f_marginal_loglik"]).all()
    assert out["f_filtered_states_mean"].shape[-2:] == (len(obs_times), 2)
    assert out["f_filtered_regime_probs"].shape[-2:] == (len(obs_times), 2)


def test_slds_rbpf_batched_predictive_trace_sites_are_finite():
    obs_times, obs_values = _make_slds_observations()
    obs_values = jnp.stack([obs_values, obs_values], axis=0)

    with Filter(
        RBPFConfig(
            n_particles=64,
            proposal="optimal",
            record_filtered_states_mean=True,
            record_filtered_regime_probs=True,
            crn_seed=jr.PRNGKey(1),
        )
    ):
        out = Predictive(
            _batched_small_slds_model,
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(2), obs_times=obs_times, obs_values=obs_values)

    assert jnp.isfinite(out["f_marginal_loglik"]).all()
    assert out["f_marginal_loglik"].shape[-1:] == (2,)


@pytest.mark.parametrize("proposal", ["prior", "optimal"])
def test_slds_rbpf_accepts_fully_missing_observation_rows(proposal):
    obs_times, obs_values = _make_slds_observations()
    obs_values = obs_values.at[2, 0].set(jnp.nan)
    obs_values = obs_values.at[5, 0].set(jnp.nan)

    with Filter(
        RBPFConfig(
            n_particles=64,
            proposal=proposal,
            record_filtered_states_mean=True,
            record_filtered_regime_probs=True,
            crn_seed=jr.PRNGKey(1),
        )
    ):
        out = Predictive(
            _small_slds_model,
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(2), obs_times=obs_times, obs_values=obs_values)

    assert jnp.isfinite(out["f_marginal_loglik"]).all()
    assert jnp.isfinite(out["f_filtered_states_mean"]).all()
    assert jnp.isfinite(out["f_filtered_regime_probs"]).all()
    assert out["f_filtered_states_mean"].shape[-2] == len(obs_times)


def test_slds_rbpf_batched_missing_observations_are_finite():
    obs_times, obs_values = _make_slds_observations()
    batched_obs = jnp.stack([obs_values, obs_values], axis=0)
    batched_obs = batched_obs.at[0, 1, 0].set(jnp.nan)
    batched_obs = batched_obs.at[1, 4, 0].set(jnp.nan)

    with Filter(
        RBPFConfig(
            n_particles=64,
            proposal="optimal",
            record_filtered_states_mean=True,
            crn_seed=jr.PRNGKey(1),
        )
    ):
        out = Predictive(
            _batched_small_slds_model,
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(2), obs_times=obs_times, obs_values=batched_obs)

    assert jnp.isfinite(out["f_marginal_loglik"]).all()
    assert out["f_marginal_loglik"].shape[-1:] == (2,)


def test_slds_rbpf_numpyro_nuts_one_iteration_smoke():
    obs_times, obs_values = _make_slds_observations()

    with Filter(
        RBPFConfig(
            n_particles=32,
            proposal="optimal",
            crn_seed=jr.PRNGKey(1),
        )
    ):
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=1,
                num_warmup=1,
                num_chains=1,
                mcmc_source="numpyro",
            ),
            model=_small_slds_model_with_parameter,
        )
        posterior_samples = inference.run(jr.PRNGKey(3), obs_times, obs_values)

    assert "bias_shift" in posterior_samples
    assert jnp.isfinite(posterior_samples["bias_shift"]).all()
