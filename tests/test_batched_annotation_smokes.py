"""Smoke tests for explicit dsx.batched plate annotations."""

import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx import DiscreteTimeSimulator
from dynestyx.inference.filter_configs import KFConfig
from dynestyx.inference.filters import Filter
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from dynestyx.models.lti_dynamics import LTI_discrete


def _vector_bias_plate_model(
    *,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    M=3,
    annotate=False,
):
    with dsx.plate("trajectories", M):
        bias_raw = numpyro.sample("bias_raw", dist.Normal(0.0, 0.2))
        bias = bias_raw[..., None]  # (M, 1) - ambiguous under heuristic
        if annotate:
            bias = dsx.batched(bias)
        dynamics = DynamicalModel(
            initial_condition=dist.MultivariateNormal(
                loc=jnp.zeros(1),
                covariance_matrix=0.1 * jnp.eye(1),
            ),
            state_evolution=LinearGaussianStateEvolution(
                A=jnp.array([[1.0]]),
                cov=0.05 * jnp.eye(1),
                bias=bias,
            ),
            observation_model=LinearGaussianObservation(
                H=jnp.array([[1.0]]),
                R=jnp.array([[0.1]]),
            ),
        )
        dsx.sample(
            "f",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            predict_times=predict_times,
        )


def _nested_partial_plate_model(
    *,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    G=2,
    M=3,
):
    state_dim = 2
    Q = 0.05 * jnp.eye(state_dim)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.2]])

    with dsx.plate("groups", G):
        # Group-level offset varies only across groups.
        beta = numpyro.sample("beta", dist.Normal(0.0, 0.15))  # (G,)
        group_bias = jnp.stack([beta, jnp.zeros_like(beta)], axis=-1)  # (G, 2)
        group_bias = dsx.batched(group_bias, plate_ndims=1)

        with dsx.plate("trajectories", M):
            alpha = numpyro.sample("alpha", dist.Uniform(0.1, 0.9))  # (M, G)
            A_base = jnp.array([[0.0, 0.1], [0.1, 0.8]])
            A = jnp.broadcast_to(A_base, (M, G, 2, 2)).copy()
            A = A.at[:, :, 0, 0].set(alpha)
            dynamics = LTI_discrete(A=A, Q=Q, H=H, R=R, b=group_bias)
            dsx.sample(
                "f",
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )


def _nested_distribution_param_model(*, predict_times=None, G=2, M=3):
    with dsx.plate("groups", G):
        loc_group = numpyro.sample("loc_group", dist.Normal(0.0, 0.2))  # (G,)
        loc = dsx.batched(loc_group[..., None], plate_ndims=1)  # (G, 1)
        with dsx.plate("trajectories", M):
            dynamics = DynamicalModel(
                initial_condition=dist.MultivariateNormal(
                    loc=loc,
                    covariance_matrix=0.1 * jnp.eye(1),
                ),
                state_evolution=LinearGaussianStateEvolution(
                    A=jnp.array([[1.0]]),
                    cov=0.05 * jnp.eye(1),
                ),
                observation_model=LinearGaussianObservation(
                    H=jnp.array([[1.0]]),
                    R=jnp.array([[0.1]]),
                ),
            )
            dsx.sample("f", dynamics, predict_times=predict_times)


def test_batched_resolves_ambiguous_vector_leaf():
    t = jnp.arange(5.0)

    with DiscreteTimeSimulator():
        with pytest.raises(
            ValueError, match="no plate-batched dynamics/data sources were found"
        ):
            with trace(), seed(rng_seed=jr.PRNGKey(0)):
                _vector_bias_plate_model(predict_times=t, M=3, annotate=False)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(1)):
            _vector_bias_plate_model(predict_times=t, M=3, annotate=True)
    assert tr["f_times"]["value"].shape == (3, 1, len(t))
    assert tr["f_states"]["value"].shape[:3] == (3, 1, len(t))
    assert tr["f_observations"]["value"].shape[:3] == (3, 1, len(t))


def test_batched_partial_nested_dims_filter_and_rollout():
    G, M = 2, 3
    t = jnp.arange(4.0)
    obs = jnp.zeros((M, G, len(t), 1))

    with DiscreteTimeSimulator():
        with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(2)):
                _nested_partial_plate_model(
                    obs_times=t,
                    obs_values=obs,
                    predict_times=t,
                    G=G,
                    M=M,
                )
    assert tr["f_marginal_loglik"]["value"].shape == (M, G)
    assert tr["f_predicted_times"]["value"].shape == (M, G, 1, len(t))
    assert tr["f_predicted_states"]["value"].shape[:4] == (M, G, 1, len(t))


def test_batched_distribution_params_inside_initial_condition():
    G, M = 2, 3
    t = jnp.arange(5.0)

    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(3)):
            _nested_distribution_param_model(predict_times=t, G=G, M=M)
    assert tr["f_times"]["value"].shape == (M, G, 1, len(t))
    assert tr["f_states"]["value"].shape[:4] == (M, G, 1, len(t))
    assert tr["f_observations"]["value"].shape[:4] == (M, G, 1, len(t))


def test_batched_api_noop_outside_plate_and_invalid_spec_errors():
    # no-op outside plate: should behave like an array in jnp operations
    x = dsx.batched(jnp.array([1.0, 2.0]))
    assert jnp.allclose(x + 1.0, jnp.array([2.0, 3.0]))

    # invalid specification is raised when plate-aware handlers consume it
    bad_bias = dsx.batched(jnp.zeros((2, 1)), plate_ndims=2)
    t = jnp.arange(3.0)

    def _bad_model(predict_times=None):
        with dsx.plate("trajectories", 2):
            dynamics = DynamicalModel(
                initial_condition=dist.MultivariateNormal(
                    loc=jnp.zeros(1),
                    covariance_matrix=0.1 * jnp.eye(1),
                ),
                state_evolution=LinearGaussianStateEvolution(
                    A=jnp.array([[1.0]]),
                    cov=0.05 * jnp.eye(1),
                    bias=bad_bias,
                ),
                observation_model=LinearGaussianObservation(
                    H=jnp.array([[1.0]]),
                    R=jnp.array([[0.1]]),
                ),
            )
            dsx.sample("f", dynamics, predict_times=predict_times)

    with DiscreteTimeSimulator():
        with pytest.raises(ValueError, match="Invalid plate_ndims"):
            with trace(), seed(rng_seed=jr.PRNGKey(4)):
                _bad_model(predict_times=t)
