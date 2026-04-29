"""
Tests for dsx.sample input matrix: obs_times, obs_values, predict_times.

Matrix of expected behavior across three handler contexts:
- Case 1: Simulator → Sample (Simulator only)
- Case 2: Simulator → Filter → Sample (Filter + Simulator)
- Case 3: Filter → Sample (Filter only)

| Input Provided | Case 1: Simulator | Case 2: Sim+Filter | Case 3: Filter |
|----------------|-------------------|--------------------|-----------------|
| obs_times, obs_values, predict_times | Solve on union | Filter consumes; Simulator runs | Runs |
| obs_times, obs_values | ODE/Discrete: runs; SDE: Error | Filter consumes; Simulator no-ops | Runs |
| obs_times, predict_times (no obs_values) | Error | Error | Error |
| predict_times only | Runs | No-op → Case 1 | No-op |
| obs_times only | Error | Error | Error |
"""

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.handlers import seed, trace

from dynestyx import DiscreteTimeSimulator, Filter, SDESimulator, Simulator
from dynestyx.inference.filter_configs import ContinuousTimeEKFConfig, EKFConfig
from tests.models import (
    jumpy_controls_model,
    jumpy_controls_model_ode,
    jumpy_controls_model_sde,
)

# Shared test data (jumpy_controls models need ctrl_times/ctrl_values)
_TIMES = jnp.arange(0.0, 2.0, 0.1)
_OBS_VALUES = jnp.ones((len(_TIMES), 1)) * 0.5  # (T, obs_dim)
_CTRL_VALUES = jnp.ones((len(_TIMES), 1)) * 10.0  # control_dim=1
_CTRL_TIMES = _TIMES


def _run_model(
    model_fn,
    obs_times=None,
    obs_values=None,
    predict_times=None,
    ctrl_times=None,
    ctrl_values=None,
    context=None,
):
    """Run model with optional context manager."""
    kwargs = {}
    if obs_times is not None:
        kwargs["obs_times"] = obs_times
    if obs_values is not None:
        kwargs["obs_values"] = obs_values
    if predict_times is not None:
        kwargs["predict_times"] = predict_times
    if ctrl_times is not None:
        kwargs["ctrl_times"] = ctrl_times
    if ctrl_values is not None:
        kwargs["ctrl_values"] = ctrl_values

    def _call():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model_fn(**kwargs)
        return tr

    if context is not None:
        with context:
            return _call()
    return _call()


# -----------------------------------------------------------------------------
# Error cases (handlers validation)
# -----------------------------------------------------------------------------


def test_error_obs_times_without_obs_values():
    """obs_times without obs_values: Error (handlers require both or neither)."""

    def model():
        return jumpy_controls_model_sde(obs_times=_TIMES, obs_values=None)

    with pytest.raises(
        ValueError, match="obs_times and obs_values must be provided together"
    ):
        model()


def test_error_obs_times_and_predict_times_without_obs_values():
    """obs_times + predict_times without obs_values: Error."""

    def model():
        return jumpy_controls_model_sde(
            obs_times=_TIMES, obs_values=None, predict_times=_TIMES
        )

    with pytest.raises(
        ValueError, match="obs_times and obs_values must be provided together"
    ):
        model()


def test_error_neither_obs_times_nor_predict_times():
    """Neither obs_times nor predict_times: Error."""

    def model():
        return jumpy_controls_model_sde()

    with pytest.raises(
        ValueError, match="At least one of obs_times or predict_times must be provided"
    ):
        model()


# -----------------------------------------------------------------------------
# Case 1: Simulator only
# -----------------------------------------------------------------------------


def test_case1_simulator_all_three_runs():
    """Case 1: obs_times + obs_values + predict_times → Simulator runs on union of times."""
    # Use DiscreteTimeSimulator (SDE rejects obs_times; ODE/Discrete accept)
    tr = _run_model(
        jumpy_controls_model,
        obs_times=_TIMES,
        obs_values=_OBS_VALUES,
        predict_times=_TIMES,
        ctrl_times=_CTRL_TIMES,
        ctrl_values=_CTRL_VALUES,
        context=DiscreteTimeSimulator(),
    )
    assert "f_times" in tr and "f_states" in tr and "f_observations" in tr


def test_case1_simulator_predict_times_only_runs():
    """Case 1: predict_times only → Simulator runs (forward simulation)."""
    tr = _run_model(
        jumpy_controls_model_sde,
        predict_times=_TIMES,
        ctrl_times=_CTRL_TIMES,
        ctrl_values=_CTRL_VALUES,
        context=SDESimulator(),
    )
    assert "f_times" in tr and "f_states" in tr and "f_observations" in tr


def test_case1_simulator_obs_times_obs_values_only_discrete_runs():
    """Case 1: obs_times + obs_values (no predict_times), DiscreteTimeSimulator → runs."""
    tr = _run_model(
        jumpy_controls_model,
        obs_times=_TIMES,
        obs_values=_OBS_VALUES,
        ctrl_times=_CTRL_TIMES,
        ctrl_values=_CTRL_VALUES,
        context=DiscreteTimeSimulator(),
    )
    assert "f_times" in tr and "f_states" in tr and "f_observations" in tr


def test_case1_simulator_obs_times_obs_values_only_ode_runs():
    """Case 1: obs_times + obs_values (no predict_times), ODESimulator → runs."""
    tr = _run_model(
        jumpy_controls_model_ode,
        obs_times=_TIMES,
        obs_values=_OBS_VALUES,
        ctrl_times=_CTRL_TIMES,
        ctrl_values=_CTRL_VALUES,
        context=Simulator(),
    )
    assert "f_times" in tr and "f_states" in tr and "f_observations" in tr


def test_case1_simulator_obs_times_obs_values_only_sde_errors():
    """Case 1: obs_times + obs_values (no predict_times), SDESimulator → Error (per matrix)."""
    # SDESimulator rejects obs_times; requires predict_times for SDE rollout
    with pytest.raises(
        ValueError, match="obs_times must not be provided|predict_times"
    ):
        _run_model(
            jumpy_controls_model_sde,
            obs_times=_TIMES,
            obs_values=_OBS_VALUES,
            ctrl_times=_CTRL_TIMES,
            ctrl_values=_CTRL_VALUES,
            context=SDESimulator(),
        )


# -----------------------------------------------------------------------------
# Case 2: Simulator + Filter
# -----------------------------------------------------------------------------


def test_case2_simulator_filter_all_three_runs():
    """Case 2: obs_times + obs_values + predict_times → Filter consumes, Simulator runs."""

    def model():
        return jumpy_controls_model_sde(
            obs_times=_TIMES,
            obs_values=_OBS_VALUES,
            predict_times=_TIMES,
            ctrl_times=_CTRL_TIMES,
            ctrl_values=_CTRL_VALUES,
        )

    with SDESimulator():
        with Filter(filter_config=ContinuousTimeEKFConfig()):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
                model()
    assert "f_marginal_loglik" in tr
    assert "f_predicted_states" in tr and "f_predicted_times" in tr


def test_case2_simulator_filter_obs_only_runs():
    """Case 2: obs_times + obs_values (no predict_times) → Filter consumes, Simulator no-ops."""

    def model():
        return jumpy_controls_model_sde(
            obs_times=_TIMES,
            obs_values=_OBS_VALUES,
            ctrl_times=_CTRL_TIMES,
            ctrl_values=_CTRL_VALUES,
        )

    with SDESimulator():
        with Filter(filter_config=ContinuousTimeEKFConfig()):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
                model()
    assert "f_marginal_loglik" in tr
    # Simulator should no-op without predict_times in this chained case.
    assert "f_times" not in tr and "f_states" not in tr and "f_observations" not in tr
    assert "f_predicted_times" not in tr and "f_predicted_states" not in tr


def test_case2_simulator_filter_predict_times_only_runs():
    """Case 2: predict_times only → Filter no-op, falls back to Case 1."""

    def model():
        return jumpy_controls_model_sde(
            predict_times=_TIMES,
            ctrl_times=_CTRL_TIMES,
            ctrl_values=_CTRL_VALUES,
        )

    with SDESimulator():
        with Filter(filter_config=ContinuousTimeEKFConfig()):
            with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
                model()
    assert "f_times" in tr and "f_states" in tr and "f_observations" in tr


# -----------------------------------------------------------------------------
# Case 3: Filter only
# -----------------------------------------------------------------------------


def test_case3_filter_all_three_runs():
    """Case 3: obs_times + obs_values + predict_times → Filter runs."""

    def model():
        return jumpy_controls_model_sde(
            obs_times=_TIMES,
            obs_values=_OBS_VALUES,
            predict_times=_TIMES,
            ctrl_times=_CTRL_TIMES,
            ctrl_values=_CTRL_VALUES,
        )

    with Filter(filter_config=ContinuousTimeEKFConfig()):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model()
    assert "f_marginal_loglik" in tr


def test_case3_filter_obs_only_runs():
    """Case 3: obs_times + obs_values (no predict_times) → Filter runs."""

    def model():
        return jumpy_controls_model_sde(
            obs_times=_TIMES,
            obs_values=_OBS_VALUES,
            ctrl_times=_CTRL_TIMES,
            ctrl_values=_CTRL_VALUES,
        )

    with Filter(filter_config=ContinuousTimeEKFConfig()):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model()
    assert "f_marginal_loglik" in tr


def test_case3_filter_predict_times_only_noop():
    """Case 3: predict_times only → Filter no-op (no obs to condition on)."""

    # Filter with no obs adds nothing; falls through. No filter output expected.
    def model():
        return jumpy_controls_model(
            predict_times=_TIMES,
            ctrl_times=_CTRL_TIMES,
            ctrl_values=_CTRL_VALUES,
        )

    with Filter(filter_config=EKFConfig()):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model()
    # No obs → filter adds nothing; no f_marginal_loglik
    assert "f_marginal_loglik" not in tr
