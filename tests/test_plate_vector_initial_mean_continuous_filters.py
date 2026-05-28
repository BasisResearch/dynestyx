"""Coverage for plate-batched continuous initial means across filter backends."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator, Discretizer, Filter, SDESimulator
from dynestyx.inference.filter_configs import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeUKFConfig,
    EKFConfig,
    EnKFConfig,
    PFConfig,
)
from dynestyx.models.lti_dynamics import LTI_continuous


def _plate_vector_initial_mean_continuous_model(
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    predict_times=None,
    M=3,
):
    state_dim = 2
    L = 0.20 * jnp.eye(state_dim)
    H = jnp.eye(state_dim)
    R = (0.08**2) * jnp.eye(state_dim)

    with dsx.plate("trajectories", M):
        alpha = numpyro.sample("alpha", dist.Uniform(0.1, 0.8))
        A_base = jnp.array([[0.0, 0.1], [-0.05, -0.6]])
        A = jnp.broadcast_to(A_base, (M, state_dim, state_dim)).copy()
        A = A.at[:, 0, 0].set(-alpha)
        mu_0_i = jnp.broadcast_to(jnp.array([0.1, 0.05]), (M, state_dim))

        dynamics = LTI_continuous(
            A=A,
            L=L,
            H=H,
            R=R,
            initial_mean=mu_0_i,
            initial_cov=0.15 * jnp.eye(state_dim),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _make_discretized_observations():
    obs_times = jnp.arange(6.0)
    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(40)):
                _plate_vector_initial_mean_continuous_model(
                    predict_times=obs_times,
                    M=3,
                )
    return obs_times, tr["f_observations"]["value"][:, 0]


def _make_continuous_observations():
    obs_times = jnp.linspace(0.0, 0.5, 6)
    with SDESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(50)):
            _plate_vector_initial_mean_continuous_model(
                predict_times=obs_times,
                M=3,
            )
    return obs_times, tr["f_observations"]["value"][:, 0]


@pytest.mark.parametrize(
    "filter_config",
    [
        pytest.param(
            EKFConfig(filter_source="cuthbert"),
            marks=pytest.mark.xfail(
                reason="Plate-batched initial means are not yet handled by cuthbert EKF after discretization.",
                strict=True,
            ),
            id="discretizer-ekf",
        ),
        pytest.param(
            EnKFConfig(
                filter_source="cuthbert",
                n_particles=8,
                crn_seed=jr.PRNGKey(41),
            ),
            marks=pytest.mark.xfail(
                reason="cuthbert EnKF still receives a plate axis inside the ensemble state for this discretized path.",
                strict=True,
            ),
            id="discretizer-enkf",
        ),
        pytest.param(
            PFConfig(
                filter_source="cuthbert",
                n_particles=16,
                crn_seed=jr.PRNGKey(41),
            ),
            marks=pytest.mark.xfail(
                reason="cuthbert PF still mismatches batched state and control shapes for this discretized path.",
                strict=True,
            ),
            id="discretizer-pf",
        ),
    ],
)
def test_plate_vector_initial_mean_continuous_discretizer_filters(filter_config):
    """Plate-batched initial means should work through discretized discrete filters."""
    obs_times, obs_values = _make_discretized_observations()

    with Filter(filter_config=filter_config):
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(42)):
                _plate_vector_initial_mean_continuous_model(
                    obs_times=obs_times,
                    obs_values=obs_values,
                    M=3,
                )

    assert tr["f_marginal_loglik"]["value"].shape == (3,)


@pytest.mark.parametrize(
    "filter_config",
    [
        pytest.param(
            ContinuousTimeEnKFConfig(
                n_particles=8,
                crn_seed=jr.PRNGKey(51),
            ),
            id="ct-enkf",
        ),
        pytest.param(ContinuousTimeEKFConfig(), id="ct-ekf"),
        pytest.param(
            ContinuousTimeDPFConfig(
                n_particles=16,
                crn_seed=jr.PRNGKey(51),
            ),
            marks=pytest.mark.xfail(
                reason="Continuous-time PF does not yet support this plate-batched initial-mean setup.",
                strict=True,
            ),
            id="ct-pf",
        ),
        pytest.param(ContinuousTimeUKFConfig(), id="ct-ukf"),
    ],
)
def test_plate_vector_initial_mean_continuous_ct_filters(filter_config):
    """Plate-batched initial means should work through continuous-time filters."""
    obs_times, obs_values = _make_continuous_observations()

    with Filter(filter_config=filter_config):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(52)):
            _plate_vector_initial_mean_continuous_model(
                obs_times=obs_times,
                obs_values=obs_values,
                M=3,
            )

    assert tr["f_marginal_loglik"]["value"].shape == (3,)
