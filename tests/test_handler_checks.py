"""Reject invalid stacks before executing inference or simulation."""

import itertools
from contextlib import ExitStack

import jax.numpy as jnp
import pytest
from effectful.ops.semantics import handler

import dynestyx as dsx
from dynestyx.control import DiscreteControlLoopSimulator
from dynestyx.handlers import _validate_handler_stack

_STAGES = [
    lambda: dsx.plate("members", 2),
    dsx.Discretizer,
    dsx.GaussianRelaxation,
    dsx.Filter,
    dsx.Simulator,
    lambda: dsx.Evaluation(dsx.ObservationScoringConfig()),
]


def _closed_loop():
    return DiscreteControlLoopSimulator(
        control_policy=lambda x_hat, t_now, t_next, s: (x_hat.mean, s)
    )


@pytest.mark.parametrize("other", [dsx.Simulator, _closed_loop])
def test_closed_loop_shares_simulation_stage(other):
    with _closed_loop(), other():
        with pytest.raises(ValueError, match="one handler per stage.*SIMULATOR"):
            _validate_handler_stack(obs_values=None, predict_times=True)


@pytest.mark.parametrize("inner", [dsx.Discretizer, dsx.Filter])
def test_closed_loop_order(inner):
    with inner(), _closed_loop():
        with pytest.raises(ValueError, match="Invalid handler order.*SIMULATOR"):
            _validate_handler_stack(obs_values=True, predict_times=True)
    with _closed_loop(), inner():
        _validate_handler_stack(
            obs_values=inner is dsx.Filter or None, predict_times=True
        )


def test_closed_loop_is_not_external_observation_inference():
    with _closed_loop():
        with pytest.raises(ValueError, match="Observations require"):
            _validate_handler_stack(obs_values=True, predict_times=True)


@pytest.mark.parametrize("factory", [dsx.Filter, dsx.Smoother, dsx.LatentPathBuilder])
def test_unused_inference_warns(factory):
    with dsx.Simulator(), factory():
        with pytest.warns(UserWarning, match="has no obs_values"):
            _validate_handler_stack(obs_values=None, predict_times=True)


@pytest.mark.parametrize("factory", [dsx.Simulator, _closed_loop])
def test_unused_simulator_warns(factory):
    with factory(), dsx.Filter():
        with pytest.warns(UserWarning, match="has no predict_times"):
            _validate_handler_stack(obs_values=True, predict_times=None)


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
