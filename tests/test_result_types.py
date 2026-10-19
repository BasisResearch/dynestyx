"""Shared result access preserves backend values and reports unsupported outputs."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

from dynestyx import (
    ConditionedResult,
    EvaluationResult,
    LatentStateResult,
    Result,
    SimulatedResult,
)


def test_conditioned_diagnostics_through_shared_result_interface():
    cumulative_loglik = jnp.array([-0.2, -0.5])
    ensemble = jnp.zeros((2, 4, 1))
    predicted_ensemble = ensemble + 1
    model_inputs = SimpleNamespace(time=jnp.arange(2.0), u=jnp.zeros((2, 0)))
    backend = SimpleNamespace(
        log_normalizing_constant=cumulative_loglik,
        ensemble=ensemble,
        predicted_ensemble=predicted_ensemble,
        model_inputs=model_inputs,
    )
    result: Result = ConditionedResult(
        states=backend, marginal_loglik=cumulative_loglik[-1]
    )

    assert result.states is backend
    assert result.log_normalizing_constant is cumulative_loglik
    assert result.ensemble is ensemble
    assert result.predicted_ensemble is predicted_ensemble
    assert result.model_inputs is model_inputs
    assert jnp.array_equal(result.model_inputs.time, jnp.arange(2.0))
    assert result.model_inputs.u.shape == (2, 0)


def test_conditioned_diagnostic_properties_support_jit_and_grad():
    def total_loglik(increments):
        backend = SimpleNamespace(log_normalizing_constant=jnp.cumsum(increments))
        result: Result = ConditionedResult(states=backend)
        return result.log_normalizing_constant[-1]

    increments = jnp.array([-0.2, -0.3])
    assert jnp.allclose(jax.jit(total_loglik)(increments), -0.5)
    assert jnp.array_equal(jax.grad(total_loglik)(increments), jnp.ones(2))


@pytest.mark.parametrize(
    "result,property_name",
    [
        (Result(), "marginal_loglik"),
        (LatentStateResult(), "log_normalizing_constant"),
        (SimulatedResult(), "joint_log_prob"),
        (EvaluationResult(), "ensemble"),
        (ConditionedResult(), "ensemble"),
        (
            ConditionedResult(states=SimpleNamespace(predicted_ensemble=None)),
            "predicted_ensemble",
        ),
    ],
)
def test_unsupported_result_properties_raise(result: Result, property_name: str):
    with pytest.raises(NotImplementedError, match=property_name):
        getattr(result, property_name)


def test_supported_but_unrecorded_fields_remain_optional():
    result: Result = ConditionedResult()
    assert result.marginal_loglik is None
    assert result.dists is None
    result = LatentStateResult()
    assert result.state_path is None
    result = SimulatedResult()
    assert result.observations is None


def test_evaluation_scores_remain_independent_mutable_dataclass_fields():
    first = EvaluationResult()
    second: Result = EvaluationResult()
    first.observation_scores["score"] = jnp.ones(2)
    assert second.observation_scores == {}

    copied = replace(first, observation_scores={"other": jnp.zeros(2)})
    assert "score" in first.observation_scores
    assert "other" not in first.observation_scores
    assert "other" in copied.observation_scores
