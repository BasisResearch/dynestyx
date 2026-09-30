"""Reject invalid stacks before executing inference or simulation."""

import itertools
from contextlib import ExitStack

import jax.numpy as jnp
import pytest
from effectful.ops.semantics import handler

import dynestyx as dsx
from dynestyx.handlers import _validate_handler_stack

_STAGES = [
    lambda: dsx.plate("members", 2),
    dsx.Discretizer,
    dsx.Filter,
    dsx.Simulator,
    lambda: dsx.Evaluation(dsx.ObservationScoringConfig()),
]


@pytest.mark.parametrize("inner,outer", list(itertools.combinations(_STAGES, 2)))
def test_reversed_stage_order(inner, outer):
    with handler(inner()), handler(outer()):
        with pytest.raises(ValueError, match="Invalid handler order.*outermost"):
            _validate_handler_stack(obs_values=None, predict_times=None)


@pytest.mark.parametrize("factory", _STAGES[1:])
def test_duplicate_stages(factory):
    with handler(factory()), handler(factory()):
        with pytest.raises(ValueError, match="one handler per stage"):
            _validate_handler_stack(obs_values=None, predict_times=None)


@pytest.mark.parametrize(
    "inner,outer",
    list(
        itertools.product([dsx.Filter, dsx.Smoother, dsx.LatentPathBuilder], repeat=2)
    ),
)
def test_duplicate_inference_before_execution(inner, outer):
    dynamics = dsx.LTI_discrete(A=jnp.eye(1), Q=jnp.eye(1), H=jnp.eye(1), R=jnp.eye(1))
    with outer(), inner():
        with pytest.raises(ValueError, match="already conditioned result"):
            dsx.sample(
                "f", dynamics, obs_times=jnp.arange(3.0), obs_values=jnp.zeros((3, 1))
            )


def test_valid_full_stack_with_nested_plates():
    with ExitStack() as stack:
        for factory in reversed([_STAGES[0], *_STAGES]):
            stack.enter_context(handler(factory()))
        _validate_handler_stack(obs_values=True, predict_times=True)


@pytest.mark.parametrize("factory", [dict, dsx.Discretizer, dsx.Simulator])
def test_observations_require_inference(factory):
    with handler(factory()):
        with pytest.raises(ValueError, match="Observations require"):
            _validate_handler_stack(obs_values=True, predict_times=None)


def test_predictions_require_simulator():
    with dsx.Filter():
        with pytest.raises(ValueError, match="predict_times requires a Simulator"):
            _validate_handler_stack(obs_values=None, predict_times=True)
