"""Custom step contracts through the public filtering entry points."""

from typing import NamedTuple, assert_type

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from jaxtyping import PRNGKeyArray
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.configs import CombineFilterConfig, KFConfig, SplitFilterConfig
from dynestyx.inference.custom_filter import FilterInputs, FilterOutputs, SplitFilter
from dynestyx.inference.observation_predictions import PredictedObservationOutputs


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


class LinearSplit(SplitFilterConfig[Belief, Belief]):
    def initialize(self, model, inputs, *, key):
        prior = model.initial_condition
        return Belief(prior.mean, prior.covariance_matrix, jnp.array(0.0))

    def predict(self, model, state, inputs, *, key):
        A, B, _, Q = model.state_evolution.params_at(inputs.time_prev, inputs.time)
        return Belief(
            A @ state.mean + B @ inputs.u_prev, A @ state.cov @ A.T + Q, state.ll
        )

    def update(self, model, state, inputs, *, key):
        S = state.cov + 0.2 * jnp.eye(1)
        gain = jnp.linalg.solve(S, state.cov.T).T
        ll = dist.MultivariateNormal(state.mean, S).log_prob(inputs.y)
        return Belief(
            state.mean + gain @ (inputs.y - state.mean),
            state.cov - gain @ state.cov,
            state.ll + ll,
        )

    def gather_outputs(self, prior: Belief, filtered: Belief) -> FilterOutputs:
        return gather(filtered)


class LinearCombine(CombineFilterConfig[FilterInputs, Belief]):
    def init_prepare(self, model, *, key):
        prior = model.initial_condition
        return Belief(prior.mean, prior.covariance_matrix, jnp.array(0.0))

    def filter_prepare(self, model, inputs, *, key):
        return inputs

    def filter_combine(self, model, state, x):
        split = LinearSplit()
        prior = jax.lax.cond(
            x.is_first_step,
            lambda: state,
            lambda: split.predict(model, state, x, key=None),
        )
        return split.update(model, prior, x, key=None)

    def gather_outputs(self, filtered: Belief) -> FilterOutputs:
        return gather(filtered)


def gather(state):
    return FilterOutputs(
        state.ll, state.mean, state.cov, extra_outputs={"squared_mean": state.mean**2}
    )


@pytest.mark.parametrize("config", [LinearSplit(), LinearCombine()])
@pytest.mark.parametrize("length", [1, 3])
def test_filter_contracts(config, length):
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
    class Filtered(NamedTuple):
        mean: jax.Array

    def noise(key: PRNGKeyArray | None) -> jax.Array:
        assert key is not None
        return jax.random.normal(key, (1,))

    class RandomWalk(SplitFilterConfig[jax.Array, Filtered]):
        def build_filter(
            self, model: dsx.DynamicalModel
        ) -> SplitFilter[jax.Array, Filtered]:
            return SplitFilter(
                lambda x, *, key: noise(key),
                lambda state, x, *, key: state.mean + noise(key),
                lambda prior, x, *, key: Filtered(prior + noise(key)),
            )

        def gather_outputs(self, prior: jax.Array, filtered: Filtered) -> FilterOutputs:
            return FilterOutputs(
                jnp.array(0.0),
                filtered.mean,
                jnp.eye(1),
                predicted_observations=PredictedObservationOutputs(
                    mean=prior, cov=jnp.eye(1)
                ),
                extra_outputs={"prior": prior},
            )

    config = RandomWalk(crn_seed=jax.random.key(5))
    assert_type(
        config.gather_outputs(jnp.zeros(1), Filtered(jnp.zeros(1))), FilterOutputs
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
    np.testing.assert_array_equal(
        a.predicted_observations.mean, a.extra_outputs["prior"]
    )
    assert a.extra_outputs["prior"].shape == (3, 2, 4, 1)
    np.testing.assert_array_equal(a.extra_outputs["prior"], b.extra_outputs["prior"])
    assert np.unique(a.extra_outputs["prior"]).size == 24


def test_parallel_custom_cuthbert_filter():
    from dynestyx.inference.integrations.cuthbert.discrete_filter import (
        build_cuthbert_filter,
    )

    class NativeKF(CombineFilterConfig):
        def build_filter(self, model):
            algorithm, _ = build_cuthbert_filter(
                model, KFConfig(filter_source="cuthbert"), None, want_parallel=True
            )
            return algorithm

        def gather_outputs(self, state):
            return FilterOutputs(
                state.log_normalizing_constant,
                state.mean,
                state.chol_cov @ state.chol_cov.T,
            )

    model = make_model(0.8)
    data = dict(obs_times=jnp.arange(3.0), obs_values=jnp.array([[0.1], [0.2], [0.3]]))
    with dsx.Filter(NativeKF(parallel=True)):
        result = dsx.condition("x", model, **data)
    with dsx.Filter(KFConfig(filter_source="cuthbert", associative=False)):
        exact = dsx.condition("x", model, **data)
    np.testing.assert_allclose(
        result.log_normalizing_constant, exact.log_normalizing_constant, atol=1e-5
    )


def test_plated_custom_particles_recording():
    particles = jnp.array([[-1.0], [1.0]])
    weights = jnp.log(jnp.array([0.25, 0.75]))

    class ParticleConfig(SplitFilterConfig[jax.Array, jax.Array]):
        def build_filter(self, model):
            return SplitFilter(
                lambda x, *, key: particles,
                lambda state, x, *, key: state,
                lambda state, x, *, key: state,
            )

        def gather_outputs(self, prior, out):
            return FilterOutputs(jnp.array(0.0), particles=out, log_weights=weights)

    config = ParticleConfig()
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


def test_missing_steps_reported_at_build_time():
    class NoUpdate(SplitFilterConfig):
        def initialize(self, model, inputs, *, key):
            return inputs.y

        def predict(self, model, previous, inputs, *, key):
            return previous

        def gather_outputs(self, prior, filtered):
            return FilterOutputs(jnp.array(0.0), filtered, jnp.eye(1))

    with pytest.raises(TypeError, match="must implement update"):
        NoUpdate().build_filter(make_model(0.8))
