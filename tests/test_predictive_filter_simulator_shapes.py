"""
Test that Predictive + Filter + Simulator returns expected shapes for
num_samples=1,4 and n_simulations=1,5 where applicable.

Uses Predictive (not trace/substitute). Covers:
- SDESimulator: num_samples=1,4; n_sim=1,5
- DiscreteTimeSimulator: num_samples=1,4; n_sim=1,5
- ODESimulator: num_samples=1,4; n_sim=1,5 (n_sim>1 skipped due to ODE vmap issues)
"""

import re
from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import pytest
from numpyro.infer import Predictive

from dynestyx import (
    DiscreteTimeSimulator,
    Filter,
    ODESimulator,
    SDESimulator,
)
from dynestyx.inference.filter_configs import (
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    KFConfig,
)
from tests.models import (
    continuous_time_stochastic_l63_model,
    discrete_time_lti_simplified_model,
    jumpy_controls_model_ode,
)


def _gen_obs_sde(source: Literal["diffrax", "em_scan"]):
    """Generate obs_times, obs_values for L63 SDE."""
    rng = jr.PRNGKey(42)
    obs_times = jnp.linspace(0.0, 1.0, 6)
    predict_times = jnp.linspace(0.0, 1.5, 10)
    with SDESimulator(n_simulations=1, source=source):
        pred = Predictive(
            continuous_time_stochastic_l63_model,
            params={"rho": jnp.array(28.0)},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(rng, predict_times=obs_times)
    obs_values = sim["f_observations"][0, 0]
    return obs_times, obs_values, predict_times, 3  # state_dim


def _gen_obs_discrete():
    """Generate obs_times, obs_values for discrete LTI."""
    rng = jr.PRNGKey(42)
    obs_times = jnp.arange(0.0, 6.0, 1.0)
    predict_times = jnp.arange(0.0, 9.0, 1.0)  # extend beyond obs for rollout
    with DiscreteTimeSimulator(n_simulations=1):
        pred = Predictive(
            discrete_time_lti_simplified_model,
            params={"alpha": jnp.array(0.35)},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(rng, predict_times=obs_times)
    obs_values = sim["f_observations"][0, 0]
    return obs_times, obs_values, predict_times, 2  # state_dim


def _gen_obs_ode():
    """Generate obs_times, obs_values for LTI ODE."""
    rng = jr.PRNGKey(42)
    obs_times = jnp.linspace(0.0, 0.5, 11)
    predict_times = jnp.linspace(0.0, 1.0, 21)
    ctrl_times = predict_times
    ctrl_values = jnp.ones((len(predict_times),)) * 100
    for i in range(1, len(ctrl_values), 2):
        ctrl_values = ctrl_values.at[i].set(-ctrl_values[i])
    with ODESimulator():
        pred = Predictive(
            jumpy_controls_model_ode,
            params={},
            num_samples=1,
            exclude_deterministic=False,
        )
        sim = pred(
            rng,
            predict_times=obs_times,
            ctrl_times=obs_times,
            ctrl_values=ctrl_values[: len(obs_times)],
        )
    obs_values = sim["f_observations"][0, 0]
    return obs_times, obs_values, predict_times, 1, ctrl_times, ctrl_values


@pytest.mark.parametrize("num_samples", [1, 2])
@pytest.mark.parametrize("n_sim", [1, 2])
@pytest.mark.parametrize("source", ["diffrax", "em_scan"])
def test_predictive_filter_sdesimulator_shapes(num_samples, n_sim, source):
    """Predictive + Filter + SDESimulator: expected shapes for num_samples and n_simulations."""
    obs_times, obs_values, predict_times, state_dim = _gen_obs_sde(source=source)

    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params={"rho": jnp.array(28.0)},
        num_samples=num_samples,
        exclude_deterministic=False,
    )
    with SDESimulator(n_simulations=n_sim, source=source):
        with Filter(
            filter_config=ContinuousTimeEnKFConfig(
                n_particles=8, record_filtered_states_mean=True
            )
        ):
            samples = predictive(
                jr.PRNGKey(0),
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    pred_states = samples["f_predicted_states"]
    pred_times = samples["f_predicted_times"]
    filtered_means = samples["f_filtered_states_mean"]

    # Always (num_samples, n_sim, T, state_dim) for consistent shaping
    expected_T = len(predict_times)
    assert pred_states.shape == (num_samples, n_sim, expected_T, state_dim)
    assert pred_times.shape == (num_samples, n_sim, expected_T)
    assert filtered_means.shape == (num_samples, len(obs_times), state_dim)


@pytest.mark.parametrize("num_samples", [1, 2])
@pytest.mark.parametrize("n_sim", [1, 2])
def test_predictive_filter_discretetimesimulator_shapes(num_samples, n_sim):
    """Predictive + Filter + DiscreteTimeSimulator: expected shapes for num_samples and n_simulations."""
    obs_times, obs_values, predict_times, state_dim = _gen_obs_discrete()

    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params={"alpha": jnp.array(0.35)},
        num_samples=num_samples,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator(n_simulations=n_sim):
        with Filter(
            filter_config=KFConfig(
                record_filtered_states_mean=True, filter_source="cuthbert"
            )
        ):
            samples = predictive(
                jr.PRNGKey(0),
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    pred_states = samples["f_predicted_states"]
    pred_times = samples["f_predicted_times"]
    filtered_means = samples["f_filtered_states_mean"]

    # Always (num_samples, n_sim, T, state_dim) for consistent shaping
    expected_T = len(predict_times)
    assert pred_states.shape == (num_samples, n_sim, expected_T, state_dim)
    assert pred_times.shape == (num_samples, n_sim, expected_T)
    assert filtered_means.shape == (num_samples, len(obs_times), state_dim)


@pytest.mark.parametrize("num_samples", [1, 2])
@pytest.mark.parametrize("n_sim", [1, 2])
def test_predictive_filter_odesimulator_shapes(num_samples, n_sim):
    """Predictive + Filter + ODESimulator: expected shapes for num_samples and n_simulations."""
    # if n_sim > 1:
    #     pytest.skip(
    #         "n_sim>1 with Predictive triggers UnexpectedTracerError in ODE vmap"
    #     )
    obs_times, obs_values, predict_times, state_dim, ctrl_times, ctrl_values = (
        _gen_obs_ode()
    )

    predictive = Predictive(
        jumpy_controls_model_ode,
        params={},
        num_samples=num_samples,
        exclude_deterministic=False,
    )
    with ODESimulator(n_simulations=n_sim):
        with Filter(
            filter_config=ContinuousTimeEKFConfig(record_filtered_states_mean=True)
        ):
            samples = predictive(
                jr.PRNGKey(0),
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
            )

    pred_states = samples["f_predicted_states"]
    pred_times = samples["f_predicted_times"]
    filtered_means = samples["f_filtered_states_mean"]

    # Always (num_samples, n_sim, T, state_dim) for consistent shaping
    assert pred_states.shape == (num_samples, n_sim, len(predict_times), state_dim)
    assert pred_times.shape == (num_samples, n_sim, len(predict_times))
    assert filtered_means.shape == (num_samples, len(obs_times), state_dim)


def test_predictive_filter_discrete_rollout_uses_only_nonempty_segments():
    """Future-only rollout should simulate only the final filtered segment.

    This is a complexity regression test: old behavior simulated every filtered
    segment (including empty ones), yielding O(n_filtered * n_pred) work.
    """
    obs_times = jnp.arange(0.0, 5.0, 1.0)  # minimal filtered states
    predict_times = jnp.arange(5.0, 8.0, 1.0)  # all strictly in the future

    # Build observed training data once.
    with DiscreteTimeSimulator(n_simulations=1):
        train_pred = Predictive(
            discrete_time_lti_simplified_model,
            params={"alpha": jnp.array(0.35)},
            num_samples=1,
            exclude_deterministic=False,
        )
        train_samples = train_pred(jr.PRNGKey(123), predict_times=obs_times)
    obs_values = train_samples["f_observations"][0, 0]

    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params={"alpha": jnp.array(0.35)},
        num_samples=2,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator(n_simulations=2):
        with Filter(
            filter_config=KFConfig(
                record_filtered_states_mean=True, filter_source="cuthbert"
            )
        ):
            samples = predictive(
                jr.PRNGKey(0),
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )

    # Segment-level simulator calls create sites like f_<segment>_x_0.
    # For future-only rollout, only one segment should be simulated.
    x0_keys = [k for k in samples if re.fullmatch(r"f_\d+_x_0", k)]
    assert len(x0_keys) == 1

    # Keep a shape assertion in the same scenario (num_samples > 1, n_sim > 1).
    assert samples["f_predicted_states"].shape == (2, 2, len(predict_times), 2)


@pytest.mark.parametrize("num_samples", [1, 2])
@pytest.mark.parametrize("n_sim", [1, 2])
def test_predictive_simulator_only_times_has_n_sim_axis_sde(num_samples, n_sim):
    """Simulator-only SDE outputs always include n_sim axis, including times."""
    predict_times = jnp.linspace(0.0, 1.0, 9)
    predictive = Predictive(
        continuous_time_stochastic_l63_model,
        params={"rho": jnp.array(28.0)},
        num_samples=num_samples,
        exclude_deterministic=False,
    )
    with SDESimulator(n_simulations=n_sim):
        samples = predictive(jr.PRNGKey(0), predict_times=predict_times)
    assert samples["f_times"].shape == (num_samples, n_sim, len(predict_times))
    assert samples["f_states"].shape[0:3] == (num_samples, n_sim, len(predict_times))
    assert samples["f_observations"].shape[0:3] == (
        num_samples,
        n_sim,
        len(predict_times),
    )


@pytest.mark.parametrize("num_samples", [1, 2])
@pytest.mark.parametrize("n_sim", [1, 2])
def test_predictive_simulator_only_times_has_n_sim_axis_discrete(num_samples, n_sim):
    """Simulator-only discrete outputs always include n_sim axis, including times."""
    predict_times = jnp.arange(0.0, 9.0, 1.0)
    predictive = Predictive(
        discrete_time_lti_simplified_model,
        params={"alpha": jnp.array(0.35)},
        num_samples=num_samples,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator(n_simulations=n_sim):
        samples = predictive(jr.PRNGKey(0), predict_times=predict_times)
    assert samples["f_times"].shape == (num_samples, n_sim, len(predict_times))
    assert samples["f_states"].shape[0:3] == (num_samples, n_sim, len(predict_times))
    assert samples["f_observations"].shape[0:3] == (
        num_samples,
        n_sim,
        len(predict_times),
    )


@pytest.mark.parametrize("num_samples", [1, 2])
@pytest.mark.parametrize("n_sim", [1, 2])
def test_predictive_simulator_only_times_has_n_sim_axis_ode(num_samples, n_sim):
    """Simulator-only ODE outputs always include n_sim axis, including times."""
    predict_times = jnp.linspace(0.0, 1.0, 21)
    ctrl_times = predict_times
    ctrl_values = jnp.ones((len(predict_times),)) * 100
    for i in range(1, len(ctrl_values), 2):
        ctrl_values = ctrl_values.at[i].set(-ctrl_values[i])
    predictive = Predictive(
        jumpy_controls_model_ode,
        params={},
        num_samples=num_samples,
        exclude_deterministic=False,
    )
    with ODESimulator(n_simulations=n_sim):
        samples = predictive(
            jr.PRNGKey(0),
            predict_times=predict_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
    assert samples["f_times"].shape == (num_samples, n_sim, len(predict_times))
    assert samples["f_states"].shape[0:3] == (num_samples, n_sim, len(predict_times))
    assert samples["f_observations"].shape[0:3] == (
        num_samples,
        n_sim,
        len(predict_times),
    )
