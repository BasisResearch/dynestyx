"""Tests for Euler–Maruyama discretization and Discretizer wrapping."""

import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.discretizers import (
    Discretizer,
    EulerMaruyamaGaussianStateEvolution,
    euler_maruyama,
)
from dynestyx.inference.filter_configs import EKFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    GaussianStateEvolution,
)
from dynestyx.models.observations import LinearGaussianObservation
from dynestyx.solvers import euler_maruyama_loc_cov


def _ctse_1d_zero_drift_unit_diffusion() -> ContinuousTimeStateEvolution:
    return ContinuousTimeStateEvolution(
        drift=lambda x, u, t: jnp.zeros_like(x),
        diffusion_coefficient=lambda x, u, t: jnp.ones((1, 1)),
        bm_dim=1,
    )


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
    x = jnp.array([[0.0, 1.0, 2.0]])  # (1, 3)
    t_now = jnp.array([0.0, 1.0, 2.0])
    t_next = jnp.array([0.5, 1.5, 2.5])
    d = evo(x, None, t_now, t_next)
    # vmap stacks the time batch on the leading axis: (num_timepoints, dim_state)
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
