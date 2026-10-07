"""Mixed-state discrete component validation in scoring and model evaluation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from dynestyx.distributions import MixedStateDistribution
from dynestyx.models import (
    SwitchingLinearGaussianObservation,
    SwitchingLinearGaussianStateEvolution,
)


@pytest.mark.parametrize("compiled", [False, True])
def test_mixed_state_log_prob_rejects_invalid_discrete_states(compiled):
    mixed = MixedStateDistribution(
        jnp.array([0.25, 0.75]),
        jnp.array([[0.0], [2.0]]),
        jnp.ones((2, 1, 1)),
    )
    discrete_states = jnp.array(
        [0.0, 1.0, -1.0, 2.0, 0.4, 1.4, jnp.nan, jnp.inf, -jnp.inf]
    )
    values = jnp.stack([discrete_states, jnp.ones_like(discrete_states)], axis=-1)
    score = jax.jit(mixed.log_prob) if compiled else mixed.log_prob
    actual = score(values)
    expected = jnp.log(jnp.array([0.25, 0.75])) + dist.Normal(
        jnp.array([0.0, 2.0]), 1.0
    ).log_prob(1.0)
    assert jnp.allclose(actual[:2], expected)
    assert jnp.isneginf(actual[2:]).all()


@pytest.mark.parametrize("rounding", [False, True])
@pytest.mark.parametrize("kind", ["transition", "observation"])
@pytest.mark.parametrize("compiled", [False, True])
def test_switching_models_validate_discrete_states(kind, compiled, rounding):
    if kind == "transition":
        transition = SwitchingLinearGaussianStateEvolution(
            transition_matrix=jnp.array([[0.9, 0.1], [0.2, 0.8]]),
            A=jnp.ones((2, 1, 1)),
            cov=jnp.ones((2, 1, 1)),
            rounding=rounding,
        )

        def evaluate_transition(state):
            return transition(state, None, 0.0, 1.0).categorical_probs

        evaluate = evaluate_transition
        expected = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    else:
        observation = SwitchingLinearGaussianObservation(
            H=jnp.array([[[1.0]], [[3.0]]]),
            R=jnp.ones((2, 1, 1)),
            rounding=rounding,
        )

        def evaluate_observation(state):
            return observation(state, None, 0.0).mean

        evaluate = evaluate_observation
        expected = jnp.array([[2.0], [6.0]])

    evaluate = eqx.filter_jit(evaluate) if compiled else evaluate
    for discrete_state in (0, 1):
        assert jnp.allclose(
            evaluate(jnp.array([float(discrete_state), 2.0])), expected[discrete_state]
        )
    if rounding:
        for value, index in [(-0.4, 0), (0.4, 0), (0.5, 0), (0.6, 1), (1.4, 1)]:
            assert jnp.allclose(evaluate(jnp.array([value, 2.0])), expected[index])
    invalid_values = [-1.0, 2.0, 1.5, float("nan"), float("inf"), -float("inf")]
    if not rounding:
        invalid_values += [0.4, 1.4]
    for discrete_state in invalid_values:
        with pytest.raises(
            eqx.EquinoxRuntimeError, match="Mixed state discrete component must be"
        ):
            evaluate(jnp.array([discrete_state, 2.0])).block_until_ready()


@pytest.mark.parametrize("compiled", [False, True])
def test_rounding_distribution_scores_rounded_states_and_preserves_option(compiled):
    transition = SwitchingLinearGaussianStateEvolution(
        transition_matrix=jnp.array([[0.9, 0.1], [0.2, 0.8]]),
        A=jnp.ones((2, 1, 1)),
        cov=jnp.ones((2, 1, 1)),
        rounding=True,
    )
    # Check propagation from the transition and NumPyro pytree reconstruction.
    mixed = transition(jnp.array([0.4, 2.0]), None, 0.0, 1.0)
    leaves, tree = jax.tree_util.tree_flatten(mixed)
    mixed = jax.tree_util.tree_unflatten(tree, leaves)
    assert mixed.rounding is True
    score = jax.jit(mixed.log_prob) if compiled else mixed.log_prob
    for value, index in [(-0.4, 0), (0.4, 0), (0.5, 0), (0.6, 1), (1.4, 1)]:
        assert jnp.allclose(
            score(jnp.array([value, 3.0])),
            jnp.log(jnp.array([0.9, 0.1])[index]) + dist.Normal(2.0, 1.0).log_prob(3.0),
        )
    for value in [-1.0, 1.5, 2.0, float("nan"), float("inf"), -float("inf")]:
        assert jnp.isneginf(score(jnp.array([value, 3.0])))
