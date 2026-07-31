import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.distributions import RaoBlackwellizedParticleDistribution
from dynestyx.inference.configs.mcmc import NUTSConfig
from dynestyx.inference.filters import KFConfig, RBPFConfig
from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (
    RBPFPosterior,
    compute_cd_dynamax_discrete_filter,
)
from dynestyx.inference.mcmc import MCMCInference


def _make_slds_dynamics(*, bias_shift=0.0):
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
    dynamics_biases = jnp.array([[0.0, 0.0], [0.75 + bias_shift, -0.25]])
    observation_matrices = jnp.tile(jnp.array([[1.0, 0.0]]), (num_regimes, 1, 1))
    observation_covariances = jnp.tile(
        0.10**2 * jnp.eye(observation_dim), (num_regimes, 1, 1)
    )

    return dsx.DynamicalModel(
        initial_condition=dsx.MixedStateDistribution(
            categorical_probs=jnp.array([0.80, 0.20]),
            continuous_locs=jnp.zeros((num_regimes, state_dim)),
            continuous_covariances=jnp.tile(jnp.eye(state_dim), (num_regimes, 1, 1)),
        ),
        state_evolution=dsx.SwitchingLinearGaussianStateEvolution(
            transition_matrix=transition_matrix,
            A=dynamics_matrices,
            cov=dynamics_covariances,
            bias=dynamics_biases,
        ),
        observation_model=dsx.SwitchingLinearGaussianObservation(
            H=observation_matrices,
            R=observation_covariances,
        ),
    )


def _make_observations():
    times = jnp.arange(8.0)
    simulation = dsx.simulate(
        _make_slds_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=times,
    )
    assert simulation.observations is not None
    return times, simulation.observations[0]


def _slds_model(
    obs_times=None,
    obs_values=None,
    *,
    sample_bias_shift=False,
):
    bias_shift = (
        numpyro.sample("bias_shift", dist.Normal(0.0, 0.1))
        if sample_bias_shift
        else 0.0
    )
    return dsx.sample(
        "f",
        _make_slds_dynamics(bias_shift=bias_shift),
        obs_times=obs_times,
        obs_values=obs_values,
    )


def test_mixed_state_distribution_samples_and_scores_batched_states():
    probs = jnp.array([[0.25, 0.75], [0.60, 0.40]])
    locs = jnp.array(
        [
            [[0.0, 1.0], [2.0, -1.0]],
            [[-1.0, 0.5], [0.25, 1.5]],
        ]
    )
    covariances = jnp.tile(jnp.eye(2)[None, None], (2, 2, 1, 1))
    mixed = dsx.MixedStateDistribution(probs, locs, covariances)

    samples = mixed.sample(jr.PRNGKey(0), sample_shape=(5,))
    value = jnp.array([[1.0, 1.7, -0.4], [0.0, -0.8, 0.1]])
    expected = dist.Categorical(probs=probs).log_prob(
        jnp.array([1, 0])
    ) + dist.MultivariateNormal(
        jnp.array([locs[0, 1], locs[1, 0]]),
        covariance_matrix=jnp.eye(2),
    ).log_prob(value[:, 1:])

    assert mixed.batch_shape == (2,)
    assert mixed.event_shape == (3,)
    assert samples.shape == (5, 2, 3)
    assert jnp.allclose(mixed.log_prob(value), expected)


def test_rbpf_distribution_retains_conditional_gaussian_uncertainty():
    posterior = RaoBlackwellizedParticleDistribution(
        log_weights=jnp.array([0.0]),
        regimes=jnp.array([1]),
        continuous_locs=jnp.zeros((1, 1)),
        continuous_covariances=jnp.ones((1, 1, 1)),
    )

    samples = posterior.sample(jr.PRNGKey(0), sample_shape=(256,))

    assert jnp.all(samples[:, 0] == 1)
    assert jnp.var(samples[:, 1]) > 0.2
    assert jnp.isfinite(posterior.log_prob(jnp.array([1.0, 0.0])))


@pytest.mark.parametrize("proposal", ["prior", "optimal"])
def test_slds_rbpf_returns_finite_typed_posterior(proposal):
    obs_times, obs_values = _make_observations()
    config = RBPFConfig(n_particles=64, proposal=proposal)
    posterior = compute_cd_dynamax_discrete_filter(
        _make_slds_dynamics(),
        config,
        key=jr.PRNGKey(1),
        obs_times=obs_times,
        obs_values=obs_values,
    )

    assert isinstance(posterior, RBPFPosterior)
    assert jnp.isfinite(posterior.marginal_loglik)
    assert posterior.weights.shape == (len(obs_times), config.n_particles)
    assert posterior.means.shape == (len(obs_times), config.n_particles, 2)
    assert posterior.filtered_means.shape == (len(obs_times), 2)
    assert posterior.filtered_covariances.shape == (len(obs_times), 2, 2)
    assert posterior.filtered_regime_probs.shape == (len(obs_times), 2)
    assert jnp.allclose(posterior.weights.sum(axis=-1), 1.0)
    assert jnp.allclose(posterior.filtered_regime_probs.sum(axis=-1), 1.0)


def test_slds_rbpf_registers_requested_trace_sites():
    obs_times, obs_values = _make_observations()
    config = RBPFConfig(
        n_particles=32,
        proposal="optimal",
        record_filtered_states_mean=True,
        record_filtered_states_cov=True,
        record_filtered_regime_probs=True,
        crn_seed=jr.PRNGKey(1),
    )

    with trace() as tr, seed(rng_seed=jr.PRNGKey(2)), dsx.Filter(config):
        _slds_model(obs_times, obs_values)

    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
    assert tr["f_filtered_states_mean"]["value"].shape == (len(obs_times), 2)
    assert tr["f_filtered_states_cov"]["value"].shape == (len(obs_times), 2, 2)
    assert tr["f_filtered_regime_probs"]["value"].shape == (len(obs_times), 2)


def test_slds_rbpf_supports_shared_model_in_plate():
    obs_times, obs_values = _make_observations()
    batched_observations = jnp.stack((obs_values, obs_values))

    def plated_model():
        with dsx.plate("trajectories", 2):
            _slds_model(obs_times, batched_observations)

    config = RBPFConfig(
        n_particles=32,
        proposal="optimal",
        crn_seed=jr.PRNGKey(1),
    )
    with trace() as tr, seed(rng_seed=jr.PRNGKey(2)), dsx.Filter(config):
        plated_model()

    marginal_loglik = tr["f_marginal_loglik"]["value"]
    assert marginal_loglik.shape == (2,)
    assert jnp.isfinite(marginal_loglik).all()


def test_slds_rbpf_supports_controls():
    obs_times, obs_values = _make_observations()
    base = _make_slds_dynamics()
    evolution = base.state_evolution
    observation = base.observation_model
    controlled = dsx.DynamicalModel(
        control_dim=1,
        initial_condition=base.initial_condition,
        state_evolution=dsx.SwitchingLinearGaussianStateEvolution(
            transition_matrix=evolution.transition_matrix,
            A=evolution.A,
            cov=evolution.cov,
            B=jnp.ones((evolution.num_regimes, 2, 1)) * 0.05,
            bias=evolution.bias,
        ),
        observation_model=dsx.SwitchingLinearGaussianObservation(
            H=observation.H,
            R=observation.R,
            D=jnp.ones((observation.num_regimes, 1, 1)) * 0.01,
            bias=observation.bias,
        ),
    )

    posterior = compute_cd_dynamax_discrete_filter(
        controlled,
        RBPFConfig(n_particles=32),
        key=jr.PRNGKey(1),
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=obs_times,
        ctrl_values=jnp.ones((len(obs_times), 1)),
    )

    assert jnp.isfinite(posterior.marginal_loglik)


def _make_degenerate_slds_and_lgssm(*, compensate_backend_initialization):
    num_regimes = 2
    state_dim = 2
    initial_mean = jnp.array([0.2, -0.1])
    initial_cov = 0.5 * jnp.eye(state_dim)
    A = jnp.array([[0.75, 0.1], [-0.05, 0.8]])
    Q = 0.05 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.3], [-0.2, 0.7]])
    R = 0.1 * jnp.eye(2)
    transition_matrix = jnp.full((num_regimes, num_regimes), 1.0 / num_regimes)

    slds = dsx.DynamicalModel(
        initial_condition=dsx.MixedStateDistribution(
            jnp.full((num_regimes,), 1.0 / num_regimes),
            jnp.tile(initial_mean[None], (num_regimes, 1)),
            jnp.tile(initial_cov[None], (num_regimes, 1, 1)),
        ),
        state_evolution=dsx.SwitchingLinearGaussianStateEvolution(
            transition_matrix,
            jnp.tile(A[None], (num_regimes, 1, 1)),
            jnp.tile(Q[None], (num_regimes, 1, 1)),
        ),
        observation_model=dsx.SwitchingLinearGaussianObservation(
            jnp.tile(H[None], (num_regimes, 1, 1)),
            jnp.tile(R[None], (num_regimes, 1, 1)),
        ),
    )

    if compensate_backend_initialization:
        # The pinned backend samples an initial Gaussian mean and then performs
        # one transition before its first observation. These are the exact
        # LGSSM moments implied by that temporary backend contract.
        initial_mean = A @ initial_mean
        initial_cov = 2.0 * A @ initial_cov @ A.T + Q
    lgssm = dsx.LTI_discrete(
        A=A,
        Q=Q,
        H=H,
        R=R,
        initial_mean=initial_mean,
        initial_cov=initial_cov,
    )
    return slds, lgssm


@pytest.mark.parametrize("proposal", ["prior", "optimal"])
def test_degenerate_slds_marginal_loglik_matches_backend_equivalent_kf(proposal):
    """Identical SLDS regimes match the KF under the pinned backend contract."""
    times = jnp.arange(6.0)
    observations = jnp.array(
        [[0.2, -0.1], [0.4, 0.0], [0.1, 0.3], [-0.2, 0.1], [0.0, -0.2], [0.3, 0.2]]
    )
    slds, lgssm = _make_degenerate_slds_and_lgssm(
        compensate_backend_initialization=True
    )

    rbpf_posterior = compute_cd_dynamax_discrete_filter(
        slds,
        RBPFConfig(n_particles=2_000, proposal=proposal),
        key=jr.PRNGKey(3),
        obs_times=times,
        obs_values=observations,
    )
    kf_posterior = compute_cd_dynamax_discrete_filter(
        lgssm,
        KFConfig(),
        obs_times=times,
        obs_values=observations,
    )

    assert jnp.allclose(
        rbpf_posterior.marginal_loglik,
        kf_posterior.marginal_loglik,
        atol=0.1,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The pinned CD-Dynamax RBPF advances the continuous state before y[0]; "
        "remove this marker when the upstream release uses the model's x[0] prior."
    ),
)
def test_degenerate_slds_marginal_loglik_matches_same_model_kf():
    """Identical regimes should reduce to the same-model Kalman filter."""
    times = jnp.arange(6.0)
    observations = jnp.array(
        [[0.2, -0.1], [0.4, 0.0], [0.1, 0.3], [-0.2, 0.1], [0.0, -0.2], [0.3, 0.2]]
    )
    slds, lgssm = _make_degenerate_slds_and_lgssm(
        compensate_backend_initialization=False
    )

    rbpf_posterior = compute_cd_dynamax_discrete_filter(
        slds,
        RBPFConfig(n_particles=4_000, proposal="optimal"),
        key=jr.PRNGKey(3),
        obs_times=times,
        obs_values=observations,
    )
    kf_posterior = compute_cd_dynamax_discrete_filter(
        lgssm,
        KFConfig(),
        obs_times=times,
        obs_values=observations,
    )

    assert jnp.allclose(
        rbpf_posterior.marginal_loglik,
        kf_posterior.marginal_loglik,
        atol=0.1,
    )


def test_slds_rbpf_numpyro_nuts_one_iteration_smoke():
    obs_times, obs_values = _make_observations()

    with dsx.Filter(
        RBPFConfig(
            n_particles=16,
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
            model=lambda obs_times, obs_values, **_: _slds_model(
                obs_times,
                obs_values,
                sample_bias_shift=True,
            ),
        )
        posterior_samples = inference.run(
            jr.PRNGKey(3),
            obs_times,
            obs_values,
        )

    assert jnp.isfinite(posterior_samples["bias_shift"]).all()
