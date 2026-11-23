"""Closed-loop control with separate plant and filter models."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed

import dynestyx as dsx
from dynestyx.control.discrete_controller_simulators import (
    ControlledSimulatedResult,
    DiscreteControlLoopSimulator,
)
from dynestyx.inference.configs.filter import EKFConfig


def _plant(
    *,
    state_dim=1,
    observation_dim=1,
    control_dim=1,
    scalar=False,
    alignment="previous_transition",
):
    initial_state = jnp.asarray(2.0) if scalar else jnp.full((state_dim,), 2.0)
    return dsx.DynamicalModel(
        initial_condition=dist.Delta(initial_state, event_dim=initial_state.ndim),
        state_evolution=dsx.DeterministicStateEvolution(
            lambda x, u, t, t_next: x + (t_next - t) * u[0]
        ),
        observation_model=dsx.DeterministicObservation(
            lambda x, u, t: jnp.broadcast_to(jnp.ravel(x)[0], (observation_dim,))
        ),
        control_dim=control_dim,
        observation_control_alignment=alignment,
    )


def _policy(x_hat, t_now, t_next, s):
    return -0.5 * x_hat.mean, s


@pytest.mark.parametrize("handler", [None, dsx.Simulator, DiscreteControlLoopSimulator])
def test_plant_trajectory_and_filter_beliefs(handler):
    plant = _plant()
    filter_model = dsx.relax_dynamics(
        plant,
        state_evolution_cov=0.05,
        observation_model_cov=0.2,
    )
    filter_model = eqx.tree_at(
        lambda model: model.initial_condition,
        filter_model,
        dist.MultivariateNormal(jnp.zeros(1), 0.4 * jnp.eye(1)),
    )
    config = EKFConfig(
        record_filtered_states_mean=True, record_filtered_states_cov=True
    )
    times = jnp.arange(4.0)
    if handler is None:
        result = dsx.simulate(
            plant,
            rng_key=jr.key(0),
            predict_times=times,
            control_policy=_policy,
            filter_dynamics=filter_model,
            filter_config=config,
        )
    else:
        with (
            seed(rng_seed=0),
            handler(
                control_policy=_policy,
                filter_dynamics=filter_model,
                filter_config=config,
            ),
        ):
            result = dsx.condition("trajectory", plant, predict_times=times)

    assert isinstance(result, ControlledSimulatedResult)
    assert result.states is not None and result.controls is not None
    assert result.observations is not None
    assert result.filtered_states_mean is not None
    assert result.filtered_states_cov is not None
    states = np.asarray(result.states[0, :, 0])
    controls = np.asarray(result.controls[0, :, 0])
    observations = np.asarray(result.observations[0, :, 0])

    assert states[0] == 2.0
    assert controls[0] == 0.0
    np.testing.assert_allclose(states[1:], states[:-1] + controls, atol=1e-7)
    np.testing.assert_array_equal(observations, states[1:])

    means, variances = [0.0], [0.4]
    for control, observation in zip(controls, observations, strict=True):
        predicted_mean = means[-1] + control
        predicted_variance = variances[-1] + 0.05
        gain = predicted_variance / (predicted_variance + 0.2)
        means.append(predicted_mean + gain * (observation - predicted_mean))
        variances.append((1 - gain) * predicted_variance)
    np.testing.assert_allclose(controls, -0.5 * np.asarray(means[:-1]), rtol=1e-6)
    np.testing.assert_allclose(result.filtered_states_mean[0, :, 0], means, rtol=1e-6)
    np.testing.assert_allclose(
        result.filtered_states_cov[0, :, 0, 0], variances, rtol=1e-6
    )


def test_omitting_filter_dynamics_matches_explicit_shared_model():
    model = dsx.relax_dynamics(
        _plant(alignment=None),
        initial_condition_cov=0.4,
        state_evolution_cov=0.05,
        observation_model_cov=0.2,
    )
    key = jr.key(0)
    times = jnp.arange(4.0)
    with pytest.warns(UserWarning, match="observation_control_alignment"):
        implicit = dsx.simulate(
            model, control_policy=_policy, rng_key=key, predict_times=times
        )
    with pytest.warns(UserWarning, match="observation_control_alignment"):
        explicit = dsx.simulate(
            model,
            filter_dynamics=model,
            control_policy=_policy,
            rng_key=key,
            predict_times=times,
        )
    jax.tree.map(np.testing.assert_array_equal, implicit, explicit)


@pytest.mark.parametrize(
    ("model_options", "message"),
    [
        ({"state_dim": 2}, "state_dim"),
        ({"observation_dim": 2}, "observation_dim"),
        ({"control_dim": 2}, "control_dim"),
        ({"scalar": True}, "initial sample shape"),
        ({"alignment": "same_time"}, "observation_control_alignment"),
    ],
)
def test_incompatible_filter_dynamics(model_options, message):
    simulator = DiscreteControlLoopSimulator(
        control_policy=_policy, filter_dynamics=_plant(**model_options)
    )
    with pytest.raises(ValueError, match=message):
        simulator.simulate(_plant(), rng_key=jr.key(0), predict_times=jnp.arange(3.0))


def test_continuous_filter_dynamics_rejected():
    plant = _plant()
    continuous_model = dsx.DynamicalModel(
        initial_condition=plant.initial_condition,
        state_evolution=dsx.DeterministicContinuousTimeStateEvolution(
            drift=lambda x, u, t: x
        ),
        observation_model=plant.observation_model,
        control_dim=plant.control_dim,
    )
    simulator = DiscreteControlLoopSimulator(
        control_policy=_policy, filter_dynamics=continuous_model
    )
    with pytest.raises(ValueError, match="filter_dynamics must be a discrete-time"):
        simulator.simulate(plant, rng_key=jr.key(0), predict_times=jnp.arange(3.0))


def test_filter_dynamics_requires_state_estimation():
    plant = _plant()
    with pytest.raises(ValueError, match="filter_dynamics requires control_policy"):
        dsx.Simulator(filter_dynamics=plant)
    with pytest.raises(
        ValueError, match="filter_dynamics requires use_true_state=False"
    ):
        DiscreteControlLoopSimulator(
            control_policy=_policy, filter_dynamics=plant, use_true_state=True
        )


def test_filter_start_time_check_survives_jit_when_returning_only_states():
    plant = _plant()
    model = dsx.relax_dynamics(
        plant,
        initial_condition_cov=0.4,
        state_evolution_cov=0.05,
        observation_model_cov=0.2,
    )

    @eqx.filter_jit
    def simulate(filter_t0):
        filter_model = eqx.tree_at(
            lambda m: m.t0, model, filter_t0, is_leaf=lambda x: x is None
        )
        return dsx.simulate(
            plant,
            filter_dynamics=filter_model,
            control_policy=_policy,
            filter_config=EKFConfig(),
            rng_key=jr.key(0),
            predict_times=jnp.arange(3.0),
        ).states

    np.testing.assert_allclose(simulate(jnp.asarray(0.0))[0, :, 0], [2.0, 1.0, 0.5])
    with pytest.raises(eqx.EquinoxRuntimeError, match=r"filter_dynamics\.t0"):
        jax.block_until_ready(simulate(jnp.asarray(1.0)))
