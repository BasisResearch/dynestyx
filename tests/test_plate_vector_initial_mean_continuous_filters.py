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
            id="discretizer-ekf",
        ),
        pytest.param(
            EnKFConfig(
                filter_source="cuthbert",
                n_particles=8,
                crn_seed=jr.PRNGKey(41),
            ),
            id="discretizer-enkf",
        ),
        pytest.param(
            PFConfig(
                filter_source="cuthbert",
                n_particles=16,
                crn_seed=jr.PRNGKey(41),
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


def _plate_vector_drift_bias_continuous_model(
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=3,
):
    """Hierarchical OU model with a plate-batched drift bias ``b = -A @ mu_i``.

    Mirrors the discretizer block of
    ``docs/tutorials/gentle_intro/08_hierarchical_inference.ipynb``: the drift
    bias is a rank-2 ``(M, state_dim)`` leaf at ``state_evolution.drift.b``. Under
    a ``Discretizer`` the bias moves to ``state_evolution.cte.drift.b``; the
    rank-1 suffix is only sliced per member because that path is whitelisted as a
    known vector field (see ``utils._is_known_vector_field``).
    """
    state_dim = 2
    A = jnp.array([[-0.8, 0.25], [-0.15, -0.6]])
    L = 0.20 * jnp.eye(state_dim)
    H = jnp.eye(state_dim)
    R = (0.08**2) * jnp.eye(state_dim)

    with dsx.plate("trajectories", M):
        mu_i = numpyro.sample(
            "mu_i", dist.MultivariateNormal(jnp.zeros(state_dim), jnp.eye(state_dim))
        )
        b = -jnp.einsum("ij,...j->...i", A, mu_i)
        dynamics = LTI_continuous(
            A=A,
            L=L,
            H=H,
            R=R,
            b=b,
            initial_mean=mu_i,
            initial_cov=0.15 * jnp.eye(state_dim),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def test_plate_vector_drift_bias_discretizer_enkf():
    """Plate-batched drift bias must be sliced per member under a discretizer.

    Regression for the hierarchical-OU discretizer block: the ``cte`` wrapper the
    discretizer inserts shifted the bias to ``state_evolution.cte.drift.b``, which
    the rank-1 whitelist did not recognise, so the ``(M, state_dim)`` bias leaked
    the plate axis into the per-member drift.
    """
    obs_times = jnp.arange(6.0)
    with DiscreteTimeSimulator():
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(60)):
                _plate_vector_drift_bias_continuous_model(predict_times=obs_times, M=3)
    obs_values = tr["f_observations"]["value"][:, 0]

    with Filter(
        filter_config=EnKFConfig(
            filter_source="cuthbert", n_particles=8, crn_seed=jr.PRNGKey(61)
        )
    ):
        with Discretizer():
            with trace() as tr, seed(rng_seed=jr.PRNGKey(62)):
                _plate_vector_drift_bias_continuous_model(
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
