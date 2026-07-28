"""Tests for continuous-time simulator config objects."""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx


@pytest.mark.parametrize(
    "simulator_cls",
    [
        dsx.Simulator,
        dsx.DiscreteTimeSimulator,
        dsx.ODESimulator,
        dsx.SDESimulator,
    ],
)
@pytest.mark.parametrize("n_simulations", [0, -1])
def test_simulators_reject_nonpositive_n_simulations(simulator_cls, n_simulations):
    with pytest.raises(
        ValueError, match="n_simulations must be greater than or equal to 1"
    ):
        simulator_cls(n_simulations=n_simulations)


def _ode_model(*, predict_times=None):
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1),
            covariance_matrix=jnp.eye(1),
        ),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.1 * x
        ),
        observation_model=lambda x, u, t: dist.MultivariateNormal(
            x, 0.2**2 * jnp.eye(1)
        ),
    )
    return dsx.sample("f", dynamics, predict_times=predict_times)


def _sde_model(*, predict_times=None):
    dynamics = dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1),
            covariance_matrix=jnp.eye(1),
        ),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.1 * x,
            diffusion=dsx.ScalarDiffusion(jnp.array(0.2), bm_dim=1),
        ),
        observation_model=lambda x, u, t: dist.MultivariateNormal(
            x, 0.2**2 * jnp.eye(1)
        ),
    )
    return dsx.sample("f", dynamics, predict_times=predict_times)


def test_odesimulator_accepts_structured_config():
    config = dsx.ODESimulatorConfig(dt0=1e-2, max_steps=10_000)
    predict_times = jnp.array([0.0, 0.5, 1.0])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.ODESimulator(simulator_config=config, n_simulations=2):
            _ode_model(predict_times=predict_times)

    assert tr["f_times"]["value"].shape == (2, len(predict_times))
    assert tr["f_states"]["value"].shape == (2, len(predict_times), 1)
    assert tr["f_observations"]["value"].shape == (2, len(predict_times), 1)


def test_sdesimulator_accepts_structured_config():
    config = dsx.SDESimulatorConfig(dt0=5e-2, source="em_scan")
    predict_times = jnp.array([0.0, 0.5, 1.0])

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.SDESimulator(simulator_config=config, n_simulations=2):
            _sde_model(predict_times=predict_times)

    assert tr["f_times"]["value"].shape == (2, len(predict_times))
    assert tr["f_states"]["value"].shape == (2, len(predict_times), 1)
    assert tr["f_observations"]["value"].shape == (2, len(predict_times), 1)


def test_simulator_routes_backend_specific_configs():
    ode_config = dsx.ODESimulatorConfig(dt0=1e-2, max_steps=10_000)
    sde_config = dsx.SDESimulatorConfig(dt0=5e-2, source="em_scan")
    predict_times = jnp.array([0.0, 0.5, 1.0])

    with trace() as ode_trace, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.Simulator(simulator_config=ode_config, n_simulations=2):
            _ode_model(predict_times=predict_times)

    with trace() as sde_trace, seed(rng_seed=jr.PRNGKey(1)):
        with dsx.Simulator(simulator_config=sde_config, n_simulations=2):
            _sde_model(predict_times=predict_times)

    assert ode_trace["f_states"]["value"].shape == (2, len(predict_times), 1)
    assert sde_trace["f_states"]["value"].shape == (2, len(predict_times), 1)


def test_simulator_rejects_mismatched_config_for_routed_backend():
    predict_times = jnp.array([0.0, 0.5, 1.0])

    with pytest.raises(ValueError, match="Pass an SDESimulatorConfig instead"):
        with trace(), seed(rng_seed=jr.PRNGKey(0)):
            with dsx.Simulator(
                simulator_config=dsx.ODESimulatorConfig(),
                n_simulations=1,
            ):
                _sde_model(predict_times=predict_times)

    with pytest.raises(ValueError, match="Pass an ODESimulatorConfig instead"):
        with trace(), seed(rng_seed=jr.PRNGKey(1)):
            with dsx.Simulator(
                simulator_config=dsx.SDESimulatorConfig(),
                n_simulations=1,
            ):
                _ode_model(predict_times=predict_times)


def test_odesimulator_uses_default_structured_config():
    simulator = dsx.ODESimulator()

    assert isinstance(simulator.simulator_config, dsx.ODESimulatorConfig)
    assert simulator.simulator_config.dt0 == 1e-3


def test_sdesimulator_uses_default_structured_config():
    simulator = dsx.SDESimulator()

    assert isinstance(simulator.simulator_config, dsx.SDESimulatorConfig)
    assert simulator.simulator_config.source == "em_scan"


def test_sdesimulator_config_rejects_tol_vbt_not_smaller_than_dt0():
    config = dsx.SDESimulatorConfig(
        dt0=1e-2,
        tol_vbt=1e-2,
        source="diffrax",
    )

    with pytest.raises(ValueError, match="tol_vbt must be smaller than dt0"):
        config.resolved_tol_vbt
