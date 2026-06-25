"""Tests for Euler–Maruyama discretization and Discretizer wrapping."""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsp_linalg
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

import dynestyx as dsx
from dynestyx.discretizers import (
    Discretizer,
    EulerMaruyamaGaussianStateEvolution,
    euler_maruyama,
    frozen_jacobian_gaussian,
    simulated_likelihood,
    taylor_moment_gaussian,
)
from dynestyx.inference.filter_configs import EKFConfig, PFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiagonalDiffusion,
    DiracIdentityObservation,
    DynamicalModel,
    FullDiffusion,
    GaussianStateEvolution,
    ScalarDiffusion,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.observations import LinearGaussianObservation
from dynestyx.solvers import euler_maruyama_loc_cov
from dynestyx.solvers.sde import _stabilize_covariance
from tests.models import continuous_time_stochastic_l63_model_dirac_obs


def _ctse_1d_zero_drift_unit_diffusion() -> StochasticContinuousTimeStateEvolution:
    cte = ContinuousTimeStateEvolution(
        drift=lambda x, u, t: jnp.zeros_like(x),
        diffusion=FullDiffusion(lambda x, u, t: jnp.ones((1, 1)), bm_dim=1),
    )
    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        state_evolution=cte,
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[1.0]]),
        ),
        control_dim=0,
    )
    assert isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution)
    return dynamics.state_evolution


def _ctse_2d_zero_drift(diffusion_form: str) -> StochasticContinuousTimeStateEvolution:
    if diffusion_form == "full":
        cte = ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.zeros_like(x),
            diffusion=FullDiffusion(jnp.eye(2), bm_dim=2),
        )
    elif diffusion_form == "diag":
        cte = ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.zeros_like(x),
            diffusion=DiagonalDiffusion(jnp.ones((2,)), bm_dim=2),
        )
    elif diffusion_form == "scalar":
        cte = ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.zeros_like(x),
            diffusion=ScalarDiffusion(jnp.array(1.0), bm_dim=2),
        )
    elif diffusion_form == "callable_full":
        cte = ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.zeros_like(x),
            diffusion=FullDiffusion(lambda x, u, t: jnp.eye(2), bm_dim=2),
        )
    elif diffusion_form == "callable_diag":
        cte = ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.zeros((2,)),
            diffusion=DiagonalDiffusion(lambda x, u, t: jnp.ones((2,)), bm_dim=2),
        )
    elif diffusion_form == "callable_scalar":
        cte = ContinuousTimeStateEvolution(
            drift=lambda x, u, t: jnp.zeros_like(x),
            diffusion=ScalarDiffusion(lambda x, u, t: jnp.array(1.0), bm_dim=2),
        )
    else:
        raise ValueError(f"Unknown diffusion form: {diffusion_form}")

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=cte,
        observation_model=LinearGaussianObservation(
            H=jnp.eye(2),
            R=jnp.eye(2),
        ),
        control_dim=0,
    )
    assert isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution)
    return dynamics.state_evolution


def test_euler_maruyama_returns_gaussian_state_evolution_with_callable_cov():
    cte = _ctse_1d_zero_drift_unit_diffusion()
    evo = euler_maruyama(cte)
    assert isinstance(evo, GaussianStateEvolution)
    assert isinstance(evo, EulerMaruyamaGaussianStateEvolution)
    assert evo.cte is cte
    assert callable(evo.cov)


def test_euler_maruyama_matches_manual_mean_and_variance():
    cte = _ctse_1d_zero_drift_unit_diffusion()
    evo = euler_maruyama(cte)
    x = jnp.array([0.4])
    t0 = jnp.array(0.0)
    t1 = jnp.array(2.0)
    d = evo(x, None, t0, t1)
    dt = float(t1 - t0)
    assert jnp.allclose(d.loc, x)
    assert jnp.allclose(d.covariance_matrix, dt * jnp.ones((1, 1)))


def test_euler_maruyama_batched_time_covariance_shape():
    cte = _ctse_1d_zero_drift_unit_diffusion()
    evo = euler_maruyama(cte)
    x = jnp.array([[0.0], [1.0], [2.0]])  # (3, 1) = (T, state_dim)
    t_now = jnp.array([0.0, 1.0, 2.0])
    t_next = jnp.array([0.5, 1.5, 2.5])
    d = evo(x, None, t_now, t_next)
    assert d.loc.shape == (3, 1)
    assert d.covariance_matrix.shape == (3, 1, 1)
    assert jnp.allclose(d.covariance_matrix[:, 0, 0], jnp.array([0.5, 0.5, 0.5]))


def test_euler_maruyama_loc_cov_single_pass_consistent_with_gaussian_state_evolution():
    cte = _ctse_1d_zero_drift_unit_diffusion()
    evo = euler_maruyama(cte)
    x = jnp.array([0.3])
    t0 = jnp.array(1.0)
    t1 = jnp.array(3.0)
    d_dict = euler_maruyama_loc_cov(cte, x, None, t0, t1)
    d = evo(x, None, t0, t1)
    assert jnp.allclose(d_dict["loc"], d.loc)
    assert jnp.allclose(d_dict["cov"], d.covariance_matrix)


def test_frozen_jacobian_gaussian_matches_ou_transition_and_improves_over_euler():
    lam = 2.0
    sigma = 0.5
    cte = StochasticContinuousTimeStateEvolution(
        drift=lambda x, u, t: -lam * x,
        diffusion=ScalarDiffusion(sigma, bm_dim=1),
    )
    x = jnp.array([1.0])
    t0 = jnp.array(0.0)
    t1 = jnp.array(1.0)
    dt = t1 - t0

    exact_loc = jnp.exp(-lam * dt) * x
    exact_cov = sigma**2 * (1.0 - jnp.exp(-2.0 * lam * dt)) / (2.0 * lam)

    fj_dist = frozen_jacobian_gaussian(cte, jitter=0.0)(x, None, t0, t1)
    em_dist = euler_maruyama(cte)(x, None, t_now=t0, t_next=t1)

    assert jnp.allclose(fj_dist.loc, exact_loc, atol=1e-6)
    assert jnp.allclose(fj_dist.covariance_matrix[0, 0], exact_cov, atol=1e-6)
    assert jnp.linalg.norm(fj_dist.loc - exact_loc) < jnp.linalg.norm(
        em_dist.loc - exact_loc
    )
    assert jnp.abs(fj_dist.covariance_matrix[0, 0] - exact_cov) < jnp.abs(
        em_dist.covariance_matrix[0, 0] - exact_cov
    )


def test_frozen_jacobian_gaussian_matches_chapter_6_affine_sde_blocks():
    F = jnp.array([[-0.4, 1.2], [-0.7, -0.3]])
    b = jnp.array([0.2, -0.1])
    L = jnp.array([[0.3, 0.1], [0.0, 0.4]])
    cte = StochasticContinuousTimeStateEvolution(
        drift=lambda x, u, t: F @ x + b,
        diffusion=FullDiffusion(L, bm_dim=2),
    )
    x = jnp.array([0.8, -0.2])
    t0 = jnp.array(0.1)
    t1 = jnp.array(0.85)
    dt = t1 - t0
    diffusion_cov = L @ L.T
    state_dim = x.shape[0]

    affine_block = jnp.zeros((state_dim + 1, state_dim + 1))
    affine_block = affine_block.at[:state_dim, :state_dim].set(F)
    affine_block = affine_block.at[:state_dim, state_dim].set(b)
    expected_loc = (
        jsp_linalg.expm(affine_block * dt) @ jnp.concatenate([x, jnp.ones(1)])
    )[:state_dim]

    cov_block = jnp.zeros((2 * state_dim, 2 * state_dim))
    cov_block = cov_block.at[:state_dim, :state_dim].set(F)
    cov_block = cov_block.at[:state_dim, state_dim:].set(diffusion_cov)
    cov_block = cov_block.at[state_dim:, state_dim:].set(-F.T)
    cov_exp = jsp_linalg.expm(cov_block * dt)
    A = cov_exp[:state_dim, :state_dim]
    expected_cov = cov_exp[:state_dim, state_dim:] @ A.T

    distn = frozen_jacobian_gaussian(cte, jitter=0.0)(x, None, t0, t1)
    assert jnp.allclose(distn.loc, expected_loc, atol=1e-6)
    assert jnp.allclose(distn.covariance_matrix, expected_cov, atol=1e-6)


def test_taylor_moment_gaussian_matches_benes_second_order_variance():
    cte = StochasticContinuousTimeStateEvolution(
        drift=lambda x, u, t: jnp.tanh(x),
        diffusion=ScalarDiffusion(1.0, bm_dim=1),
    )
    x = jnp.array([0.5])
    t0 = jnp.array(0.0)
    t1 = jnp.array(0.4)
    dt = t1 - t0
    drift = jnp.tanh(x[0])

    taylor_dist = taylor_moment_gaussian(cte, jitter=0.0)(x, None, t0, t1)
    em_dist = euler_maruyama(cte)(x, None, t_now=t0, t_next=t1)

    exact_loc = x + drift * dt
    exact_var = dt + (1.0 - drift**2) * dt**2

    assert jnp.allclose(taylor_dist.loc, exact_loc, atol=1e-5)
    assert jnp.allclose(taylor_dist.covariance_matrix[0, 0], exact_var, atol=1e-5)
    assert jnp.abs(taylor_dist.covariance_matrix[0, 0] - exact_var) < jnp.abs(
        em_dist.covariance_matrix[0, 0] - exact_var
    )


def test_covariance_stabilizer_clips_negative_eigenvalues_without_global_inflation():
    cov = jnp.diag(jnp.array([-100.0, 2.0]))

    stabilized = _stabilize_covariance(cov, jitter=1e-6)
    eigvals = jnp.linalg.eigvalsh(stabilized)

    assert jnp.all(eigvals >= 1e-6)
    assert jnp.allclose(eigvals, jnp.array([1e-6, 2.0]), atol=1e-8)


def test_new_gaussian_discretizers_batched_shapes_and_positive_covariance():
    cte = _ctse_1d_zero_drift_unit_diffusion()
    x = jnp.array([[0.0], [1.0], [2.0]])
    t_now = jnp.array([0.0, 1.0, 2.0])
    t_next = jnp.array([0.5, 1.5, 2.5])

    for discretize in (frozen_jacobian_gaussian, taylor_moment_gaussian):
        d = discretize(cte)(x, None, t_now=t_now, t_next=t_next)
        assert d.loc.shape == (3, 1)
        assert d.covariance_matrix.shape == (3, 1, 1)
        assert jnp.all(jnp.linalg.eigvalsh(d.covariance_matrix) > 0.0)


def test_simulated_likelihood_mixture_log_prob_sample_and_crn_reproducibility():
    cte = StochasticContinuousTimeStateEvolution(
        drift=lambda x, u, t: -0.5 * x,
        diffusion=ScalarDiffusion(1.0, bm_dim=1),
    )
    x = jnp.array([0.4])
    kwargs: dict[str, Any] = dict(n_substeps=3, n_simulations=8, seed=123, jitter=1e-6)
    d1 = simulated_likelihood(cte, **kwargs)(
        x, None, t_now=jnp.array(0.0), t_next=jnp.array(1.0)
    )
    d2 = simulated_likelihood(cte, **kwargs)(x, None, jnp.array(0.0), jnp.array(1.0))
    d3 = simulated_likelihood(cte, **{**kwargs, "seed": 321})(
        x, None, jnp.array(0.0), jnp.array(1.0)
    )

    assert isinstance(d1, dist.MixtureSameFamily)
    assert d1.event_shape == (1,)
    assert d1.component_distribution.loc.shape == (8, 1)
    assert jnp.allclose(d1.component_distribution.loc, d2.component_distribution.loc)
    assert not jnp.allclose(
        d1.component_distribution.loc, d3.component_distribution.loc
    )
    assert jnp.isfinite(d1.log_prob(jnp.array([0.1])))
    assert d1.sample(jr.PRNGKey(1)).shape == (1,)


@pytest.mark.parametrize(
    "diffusion_form",
    ["full", "diag", "scalar", "callable_full", "callable_diag", "callable_scalar"],
)
def test_euler_maruyama_structured_diffusions_match_dense_covariance(diffusion_form):
    cte = _ctse_2d_zero_drift(diffusion_form)
    x = jnp.array([0.3, -0.1])
    t0 = jnp.array(1.0)
    t1 = jnp.array(3.0)
    out = euler_maruyama_loc_cov(cte, x, None, t0, t1)
    expected_cov = (t1 - t0) * jnp.eye(2)
    assert jnp.allclose(out["loc"], x)
    assert jnp.allclose(out["cov"], expected_cov)


def test_discretized_dirac_observations_preserve_state_dimension():
    obs_times = jnp.arange(0.0, 0.05, 0.01)
    predictive = Predictive(
        continuous_time_stochastic_l63_model_dirac_obs,
        params={"rho": jnp.array(28.0)},
        num_samples=1,
        exclude_deterministic=False,
    )

    with dsx.DiscreteTimeSimulator():
        with Discretizer():
            samples = predictive(jr.PRNGKey(0), predict_times=obs_times)

    assert samples["f_states"].shape[-1] == 3
    assert samples["f_observations"].shape[-1] == 3
    assert jnp.allclose(samples["f_states"], samples["f_observations"])


def test_dirac_identity_observation_preserves_scalar_event_shape():
    obs = DiracIdentityObservation()(jnp.array(1.23), None, jnp.array(0.0))
    assert obs.batch_shape == ()
    assert obs.event_shape == ()


def test_discretized_gaussian_state_evolution_ekf_cuthbert_smoke():
    """Callable cov + cuthbert EKF should run without cd_dynamax."""
    obs_times = jnp.arange(4.0)
    obs_values = jnp.zeros((4, 1))

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1), covariance_matrix=jnp.eye(1)
        ),
        state_evolution=_ctse_1d_zero_drift_unit_diffusion(),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]), R=jnp.array([[0.25]])
        ),
        control_dim=0,
    )

    def model():
        return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
        with Discretizer():
            with trace() as tr, seed(rng_seed=1):
                model()

    assert "f_marginal_loglik" in tr


def test_frozen_jacobian_gaussian_and_taylor_moment_ekf_cuthbert_smoke():
    obs_times = jnp.arange(4.0)
    obs_values = jnp.zeros((4, 1))

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1), covariance_matrix=jnp.eye(1)
        ),
        state_evolution=_ctse_1d_zero_drift_unit_diffusion(),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]), R=jnp.array([[0.25]])
        ),
        control_dim=0,
    )

    def model():
        return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    for discretize in (frozen_jacobian_gaussian, taylor_moment_gaussian):
        with Filter(filter_config=EKFConfig(filter_source="cuthbert")):
            with Discretizer(discretize=discretize):
                with trace() as tr, seed(rng_seed=1):
                    model()
        assert "f_marginal_loglik" in tr


def test_simulated_likelihood_pf_cuthbert_smoke():
    obs_times = jnp.arange(4.0)
    obs_values = jnp.zeros((4, 1))

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1), covariance_matrix=jnp.eye(1)
        ),
        state_evolution=_ctse_1d_zero_drift_unit_diffusion(),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]), R=jnp.array([[0.25]])
        ),
        control_dim=0,
    )

    def model():
        return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)

    def discretize(cte):
        return simulated_likelihood(
            cte,
            n_substeps=2,
            n_simulations=4,
            seed=0,
            jitter=1e-6,
        )

    with Filter(filter_config=PFConfig(n_particles=16, crn_seed=jr.PRNGKey(0))):
        with Discretizer(discretize=discretize):
            with trace() as tr, seed(rng_seed=1):
                model()

    assert "f_marginal_loglik" in tr
