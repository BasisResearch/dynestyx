from typing import Any, cast

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    FullDiffusion,
    LinearGaussianObservation,
)
from tests.models import (
    continuous_time_stochastic_l63_model,
    discrete_time_lti_simplified_model,
)


def _make_discrete_dynamics():
    return dsx.LTI_discrete(
        A=jnp.array([[0.8, 0.1], [0.1, 0.7]]),
        Q=0.05 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.2**2]]),
    )


def _make_ode_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1), covariance_matrix=jnp.eye(1)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.5 * x,
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.1**2]]),
        ),
    )


def _make_sde_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1), covariance_matrix=jnp.eye(1)
        ),
        state_evolution=ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.5 * x,
            diffusion=FullDiffusion(jnp.eye(1)),
        ),
        observation_model=LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.1**2]]),
        ),
    )


def test_simulate_accepts_high_level_n_simulations_for_discrete():
    times = jnp.arange(5.0)
    sim = dsx.simulate(
        _make_discrete_dynamics(),
        predict_times=times,
        rng_key=jr.PRNGKey(0),
        n_simulations=2,
    )
    assert sim.times.shape == (2, len(times))
    assert sim.states.shape == (2, len(times), 2)
    assert sim.observations.shape == (2, len(times), 1)


def test_simulate_accepts_concrete_continuous_configs():
    times = jnp.linspace(0.0, 0.5, 6)

    ode_sim = dsx.simulate(
        _make_ode_dynamics(),
        predict_times=times,
        rng_key=jr.PRNGKey(1),
        n_simulations=2,
        simulator_config=dsx.ODESimulatorConfig(dt0=5e-2),
    )
    assert ode_sim.states.shape == (2, len(times), 1)

    sde_sim = dsx.simulate(
        _make_sde_dynamics(),
        predict_times=times,
        rng_key=jr.PRNGKey(2),
        n_simulations=2,
        simulator_config=dsx.SDESimulatorConfig(
            dt0=5e-2,
            source="em_scan",
        ),
    )
    assert sde_sim.states.shape == (2, len(times), 1)


def test_simulate_rejects_mismatched_config():
    times = jnp.arange(4.0)
    with pytest.raises(ValueError, match="incompatible"):
        dsx.simulate(
            _make_discrete_dynamics(),
            predict_times=times,
            rng_key=jr.PRNGKey(3),
            simulator_config=dsx.SDESimulatorConfig(),
        )


def test_simulator_routes_from_config():
    times = jnp.linspace(0.0, 0.2, 4)
    with dsx.Simulator(
        n_simulations=2,
        simulator_config=dsx.SDESimulatorConfig(
            source="em_scan",
            dt0=5e-2,
        ),
    ):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(4)):
            continuous_time_stochastic_l63_model(predict_times=times)
    assert tr["f_states"]["value"].shape == (2, len(times), 3)


def test_simulator_rejects_mismatched_config():
    times = jnp.arange(4.0)
    with dsx.Simulator(simulator_config=dsx.SDESimulatorConfig()):
        with pytest.raises(ValueError, match="incompatible"):
            with trace(), seed(rng_seed=jr.PRNGKey(5)):
                discrete_time_lti_simplified_model(predict_times=times)


def test_simulate_legacy_kwargs_not_supported():
    times = jnp.arange(4.0)
    simulate_any = cast(Any, dsx.simulate)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        simulate_any(
            _make_discrete_dynamics(),
            predict_times=times,
            rng_key=jr.PRNGKey(6),
            dt0=1e-2,
        )


def test_simulator_legacy_kwargs_not_supported():
    simulator_ctor = cast(Any, dsx.Simulator)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        simulator_ctor(dt0=1e-2)
