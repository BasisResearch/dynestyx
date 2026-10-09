"""Continuous-time IC coercion preserves densities and respects explicit dimensions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.inference.configs.filter import ContinuousTimeEKFConfig
from dynestyx.inference.integrations.cd_dynamax.utils import gaussian_to_nlgssm_params
from dynestyx.models.checkers import _coerce_initial_condition


def _continuous(initial, state_dim=None, stochastic=False):
    return dsx.DynamicalModel(
        initial_condition=initial,
        state_dim=state_dim,
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -x,
            diffusion=dsx.ScalarDiffusion(0.1, bm_dim=1) if stochastic else None,
        ),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.ones((1, state_dim or 1)), R=jnp.eye(1) * 0.1
        ),
    )


@pytest.mark.parametrize("factory", [dist.Normal, dist.Uniform, dist.LogNormal])
def test_scalar_ic_preserves_density_moments_and_gradients(factory):
    base = factory(0.2, 1.0)
    promoted = _continuous(base).initial_condition
    assert promoted.batch_shape == ()
    assert promoted.event_shape == (1,)
    samples = promoted.sample(jr.key(0), (5,))
    assert samples.shape == (5, 1)
    assert jnp.allclose(promoted.mean[0], base.mean)
    assert jnp.allclose(promoted.variance[0], base.variance)
    assert jnp.allclose(promoted.log_prob(samples), base.log_prob(samples[:, 0]))
    derivative = jax.jit(
        jax.grad(
            lambda loc: _continuous(factory(loc, 1.0)).initial_condition.mean.sum()
        )
    )(jnp.array(0.2))
    expected = jax.grad(lambda loc: factory(loc, 1.0).mean)(jnp.array(0.2))
    assert jnp.allclose(derivative, expected)


@pytest.mark.parametrize("stochastic", [False, True])
def test_promoted_normal_supports_filter_and_posterior_prediction(stochastic):
    model = _continuous(dist.Normal(0.0, 1.0), stochastic=stochastic)
    simulator = dsx.SDESimulator if stochastic else dsx.ODESimulator
    with (
        simulator(n_simulations=2),
        dsx.Filter(filter_config=ContinuousTimeEKFConfig()),
        seed(rng_seed=0),
        trace() as tr,
    ):
        dsx.sample(
            "f",
            model,
            obs_times=jnp.array([0.0, 0.1]),
            obs_values=jnp.array([[0.2], [0.1]]),
            predict_times=jnp.array([0.0, 0.05, 0.1, 0.2]),
        )
    assert jnp.isfinite(tr["f_marginal_loglik"]["value"])
    assert tr["f_predicted_states"]["value"].shape == (2, 4, 1)
    assert tr["f_predicted_observations"]["value"].shape == (2, 4, 1)


def test_vector_ic_preserved_and_explicit_dimension_validated():
    initial = dist.MultivariateNormal(jnp.zeros(3), jnp.eye(3))
    assert _continuous(initial, state_dim=3).initial_condition is initial
    for initial in [initial, dist.Normal(0.0, 1.0), dist.Normal(jnp.zeros(3), 1.0)]:
        with pytest.raises(ValueError, match="state_dim"):
            _continuous(initial, state_dim=2)
    model = _continuous(dist.Normal(jnp.zeros(3), 1.0), state_dim=3)
    assert model.initial_condition.batch_shape == ()
    assert model.initial_condition.event_shape == (3,)


@pytest.mark.parametrize("state_dim", [1, 3])
def test_promoted_normal_maps_to_diagonal_backend_covariance(state_dim):
    means = jnp.arange(float(state_dim))
    scales = jnp.arange(1.0, state_dim + 1.0)
    initial = _continuous(dist.Normal(means, scales), state_dim).initial_condition
    model = dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=dsx.LinearGaussianStateEvolution(
            A=jnp.eye(state_dim), cov=jnp.eye(state_dim) * 0.1
        ),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.eye(state_dim), R=jnp.eye(state_dim) * 0.1
        ),
    )
    params = gaussian_to_nlgssm_params(model)
    assert jnp.array_equal(params.initial_mean, means)
    assert jnp.array_equal(params.initial_covariance, jnp.diag(scales**2))


@pytest.mark.parametrize("members", [1, 3])
def test_expanded_normal_preserves_plate_members(members):
    initial = dist.Normal(0.0, 1.0).expand((members,))
    with dsx.plate("members", members):
        model = _continuous(initial, state_dim=1)
    assert model.initial_condition.batch_shape == (members,)
    assert model.initial_condition.event_shape == (1,)
    values = initial.sample(jr.key(1), (4,))
    assert jnp.allclose(
        model.initial_condition.log_prob(values[..., None]), initial.log_prob(values)
    )


def test_existing_singleton_coordinate_axis_is_not_duplicated_under_plate():
    with dsx.plate("members", 3):
        model = _continuous(dist.Normal(jnp.zeros((3, 1)), 1.0), state_dim=1)
    assert model.initial_condition.batch_shape == (3,)
    assert model.initial_condition.event_shape == (1,)


def test_nested_singleton_plate_is_not_consumed_as_event():
    with dsx.plate("outer", 3), dsx.plate("inner", 1):
        model = _continuous(dist.Normal(jnp.zeros((3, 1)), 1.0), state_dim=1)
    assert model.initial_condition.batch_shape == (3, 1)
    assert model.initial_condition.event_shape == (1,)


def test_ambiguous_batch_and_unsupported_conversion_warn_without_mutation():
    initial = dist.Normal(jnp.zeros(3), 1.0)
    with pytest.warns(UserWarning, match="Specify state_dim explicitly"):
        promoted = _coerce_initial_condition(initial, None)
    assert promoted is initial
    with dsx.plate("members", 3):
        initial = dist.Uniform(jnp.zeros(3), jnp.ones(3))
        with pytest.warns(UserWarning, match="batched Normal"):
            model = _continuous(initial, state_dim=1)
        assert model.initial_condition is initial


def test_categorical_ic_is_not_coerced():
    initial = dist.Categorical(probs=jnp.array([0.4, 0.6]))
    assert _coerce_initial_condition(initial, None) is initial
