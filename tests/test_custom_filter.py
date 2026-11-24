"""Custom step contracts through the public filtering entry points."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from cuthbert.inference import Filter as Fused
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.configs import CustomFilterConfig, KFConfig
from dynestyx.inference.custom_filter import FilterOutputs, SplitFilter


class Belief(NamedTuple):
    mean: jax.Array
    cov: jax.Array
    ll: jax.Array


def make_model(a):
    return dsx.LTI_discrete(
        A=jnp.array([[a]]),
        Q=jnp.array([[0.1]]),
        H=jnp.eye(1),
        R=0.2 * jnp.eye(1),
        B=jnp.eye(1),
    )


def algorithm(model):
    def initialize(inputs, *, key):
        prior = model.initial_condition
        return Belief(prior.mean, prior.covariance_matrix, jnp.array(0.0))

    def predict(state, inputs, *, key):
        A, B, _, Q = model.state_evolution.params_at(inputs.time_prev, inputs.time)
        return Belief(
            A @ state.mean + B @ inputs.u_prev, A @ state.cov @ A.T + Q, state.ll
        )

    def update(state, inputs, *, key):
        S = state.cov + 0.2 * jnp.eye(1)
        gain = jnp.linalg.solve(S, state.cov.T).T
        ll = dist.MultivariateNormal(state.mean, S).log_prob(inputs.y)
        return Belief(
            state.mean + gain @ (inputs.y - state.mean),
            state.cov - gain @ state.cov,
            state.ll + ll,
        )

    return SplitFilter(initialize, predict, update)


def fused(model):
    split = algorithm(model)

    def combine(state, x):
        prior = jax.lax.cond(
            x.is_first_step, lambda: state, lambda: split.predict(state, x, key=None)
        )
        return split.update(prior, x, key=None)

    return Fused(
        lambda *, key: split.initialize(None, key=key),
        lambda x, *, key: x,
        combine,
        False,
    )


def gather(state):
    return FilterOutputs(
        state.ll, state.mean, state.cov, extra_outputs={"squared_mean": state.mean**2}
    )


@pytest.mark.parametrize("factory", [algorithm, fused])
@pytest.mark.parametrize("length", [1, 3])
def test_filter_contracts(factory, length):
    config = CustomFilterConfig(
        build_filter=factory,
        gather_outputs=(lambda prior, state: gather(state))
        if factory is algorithm
        else gather,
    )
    data = dict(
        obs_times=jnp.array([1.0, 2.0, 4.0])[:length],
        obs_values=jnp.array([[0.1], [0.8], [-0.3]])[:length],
        ctrl_times=jnp.array([1.0, 2.0, 4.0])[:length],
        ctrl_values=jnp.array([[1.0], [-0.2], [0.0]])[:length],
    )

    def condition(a, config):
        with dsx.Filter(config):
            return dsx.condition("x", make_model(a), **data)

    with trace() as sites:
        result = condition(0.8, config)
    assert not sites
    reference = condition(0.8, KFConfig(filter_source="cuthbert"))
    for got, expected in zip(result.dists, reference.dists, strict=True):
        np.testing.assert_allclose(got.mean, expected.mean, atol=1e-5)
        np.testing.assert_allclose(
            got.covariance_matrix, expected.covariance_matrix, atol=1e-5
        )
    np.testing.assert_allclose(
        result.log_normalizing_constant, reference.log_normalizing_constant, atol=1e-5
    )
    np.testing.assert_allclose(
        result.extra_outputs["squared_mean"], reference.states.mean**2, atol=1e-5
    )
    if length > 1:
        grad = jax.grad(lambda a: condition(a, config).marginal_loglik)(0.8)
        ref_grad = jax.grad(
            lambda a: condition(a, KFConfig(filter_source="cuthbert")).marginal_loglik
        )(0.8)
        np.testing.assert_allclose(grad, ref_grad, atol=1e-5)
    with trace() as sites, seed(rng_seed=0), dsx.Filter(config):
        dsx.sample("x", make_model(0.8), **data)
    assert [s for s in sites if sites[s]["type"] == "sample" and "likelihood" in s] == [
        "x_marginal_log_likelihood"
    ]


def test_keys_and_nested_plates():
    def build(model):
        def update(prior, x, *, key):
            return prior + jax.random.normal(key, (1,))

        return SplitFilter(
            lambda x, *, key: jax.random.normal(key, (1,)),
            lambda state, x, *, key: state + jax.random.normal(key, (1,)),
            update,
        )

    config = CustomFilterConfig(
        build_filter=build,
        crn_seed=jax.random.key(5),
        gather_outputs=lambda prior, filtered: FilterOutputs(
            jnp.array(0.0), filtered, jnp.eye(1), extra_outputs={"prior": prior}
        ),
    )

    def run():
        with dsx.Filter(config), dsx.plate("outer", 2), dsx.plate("inner", 3):
            return dsx.condition(
                "x",
                make_model(0.8),
                obs_times=jnp.arange(4.0),
                obs_values=jnp.zeros((3, 2, 4, 1)),
            )

    a, b = run(), run()
    assert a.extra_outputs["prior"].shape == (3, 2, 4, 1)
    np.testing.assert_array_equal(a.extra_outputs["prior"], b.extra_outputs["prior"])
    assert np.unique(a.extra_outputs["prior"]).size == 24


def test_parallel_custom_cuthbert_filter():
    from dynestyx.inference.integrations.cuthbert.discrete_filter import (
        build_cuthbert_filter,
    )

    def build(model):
        algorithm, _ = build_cuthbert_filter(
            model, KFConfig(filter_source="cuthbert"), None, want_parallel=True
        )
        return algorithm

    def collect(state):
        return FilterOutputs(
            state.log_normalizing_constant,
            state.mean,
            state.chol_cov @ state.chol_cov.T,
        )

    model = make_model(0.8)
    data = dict(obs_times=jnp.arange(3.0), obs_values=jnp.array([[0.1], [0.2], [0.3]]))
    with dsx.Filter(
        CustomFilterConfig(build_filter=build, gather_outputs=collect, parallel=True)
    ):
        result = dsx.condition("x", model, **data)
    with dsx.Filter(KFConfig(filter_source="cuthbert", associative=False)):
        exact = dsx.condition("x", model, **data)
    np.testing.assert_allclose(
        result.log_normalizing_constant, exact.log_normalizing_constant, atol=1e-5
    )


def test_plated_custom_particles_recording():
    particles = jnp.array([[-1.0], [1.0]])
    weights = jnp.log(jnp.array([0.25, 0.75]))

    def build(model):
        return SplitFilter(
            lambda x, *, key: particles,
            lambda state, x, *, key: state,
            lambda state, x, *, key: state,
        )

    config = CustomFilterConfig(
        build_filter=build,
        gather_outputs=lambda prior, out: FilterOutputs(
            jnp.array(0.0), particles=out, log_weights=weights
        ),
    )
    with (
        trace() as sites,
        seed(rng_seed=0),
        dsx.Filter(config),
        dsx.plate("members", 2),
    ):
        dsx.sample(
            "x",
            make_model(0.8),
            obs_times=jnp.arange(3.0),
            obs_values=jnp.zeros((2, 3, 1)),
        )
    np.testing.assert_allclose(
        sites["x_filtered_states_mean"]["value"], jnp.full((2, 3, 1), 0.5), atol=1e-6
    )
    np.testing.assert_allclose(
        sites["x_filtered_states_cov_diag"]["value"],
        jnp.full((2, 3, 1), 0.75),
        atol=1e-6,
    )
