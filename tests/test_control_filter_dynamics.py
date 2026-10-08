"""Closed-loop control with separate environment and state-estimation models."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed

import dynestyx as dsx
from dynestyx.control.discrete_controller_simulators import DiscreteControlLoopSimulator
from dynestyx.inference.configs.filter import EKFConfig


def test_closed_loop_relaxed_filter_unrelaxed_simulation_example():
    """Mini-example: estimate with Gaussian noise, simulate an exact integrator."""
    # The environment is fully deterministic: x_next = x + dt*u, y = x.
    dynamics = dsx.DynamicalModel(
        initial_condition=dist.Delta(jnp.array([2.0]), event_dim=1),
        state_evolution=dsx.DeterministicStateEvolution(
            lambda x, u, t, tn: x + (tn - t) * u
        ),
        observation_model=dsx.DiracIdentityObservation(),
        control_dim=1,
        observation_control_alignment="previous_transition",
    )
    # These are variances, used only by the estimator.
    filter_dynamics = dsx.relax_dynamics(
        dynamics,
        initial_condition_cov=0.4,
        state_evolution_cov=0.05,
        observation_model_cov=0.2,
    )

    def policy(x_hat, t_now, t_next, s):
        # Steer toward zero using the estimated mean. Record the variance so
        # the test can check the uncertainty actually seen by the controller.
        return -0.5 * x_hat.mean, x_hat.variance

    result = dsx.simulate(
        dynamics,
        filter_dynamics=filter_dynamics,
        control_policy=policy,
        filter_config=EKFConfig(record_filtered_states_mean=True),
        rng_key=jr.key(0),
        predict_times=jnp.arange(6.0),
        initial_policy_state=jnp.zeros(1),
    )

    # Relaxation must never inject noise into the true initial state,
    # environment transitions, or generated observations.
    assert result.states is not None and result.controls is not None
    assert result.observations is not None and result.filtered_states_mean is not None
    states, controls = result.states[0], result.controls[0]
    np.testing.assert_array_equal(states[0], [2.0])
    np.testing.assert_allclose(states[1:], states[:-1] + controls, atol=1e-7)
    np.testing.assert_array_equal(result.observations[0], states[1:])
    np.testing.assert_allclose(states[:, 0], 2.0 * 0.5 ** np.arange(6), atol=1e-6)
    np.testing.assert_allclose(result.filtered_states_mean[0], states, atol=1e-6)

    # The scalar Kalman covariance recursion independently checks all three
    # relaxed variances, including the prior used by the first policy call.
    expected_variances = [0.4]
    for _ in range(len(controls) - 1):
        predicted_variance = expected_variances[-1] + 0.05
        expected_variances.append(predicted_variance * 0.2 / (predicted_variance + 0.2))
    np.testing.assert_allclose(
        np.asarray(result.policy_states).ravel(), expected_variances, rtol=1e-5
    )


def _environment(*, state_dim=1, observation_dim=1, control_dim=1, scalar=False):
    initial = jnp.asarray(2.0) if scalar else jnp.full((state_dim,), 2.0)
    return dsx.DynamicalModel(
        initial_condition=dist.Delta(initial, event_dim=0 if scalar else 1),
        state_evolution=dsx.DeterministicStateEvolution(lambda x, u, t, tn: x + u[0]),
        observation_model=dsx.DeterministicObservation(
            lambda x, u, t: jnp.broadcast_to(jnp.ravel(x)[0], (observation_dim,))
        ),
        control_dim=control_dim,
        observation_control_alignment="previous_transition",
    )


def _relaxed(dynamics):
    return dsx.relax_dynamics(
        dynamics,
        initial_condition_cov=0.4,
        state_evolution_cov=0.05,
        observation_model_cov=0.2,
    )


def _policy(x_hat, t_now, t_next, s):
    return -0.5 * x_hat.mean, s


@pytest.mark.parametrize(
    "simulator_class", [dsx.Simulator, DiscreteControlLoopSimulator]
)
def test_filter_dynamics_through_simulator_handler(simulator_class):
    """Handler-based simulation also keeps the filter prior separate."""
    dynamics = _environment()
    # Deliberately give the estimator an incorrect initial mean. The first
    # control must use that belief, while the environment starts at x=2.
    filter_dynamics = eqx.tree_at(
        lambda m: m.initial_condition,
        _relaxed(dynamics),
        dist.MultivariateNormal(jnp.zeros(1), 0.4 * jnp.eye(1)),
    )
    with (
        seed(rng_seed=0),
        simulator_class(
            control_policy=_policy,
            filter_config=EKFConfig(record_filtered_states_mean=True),
            filter_dynamics=filter_dynamics,
        ),
    ):
        result = dsx.condition("trajectory", dynamics, predict_times=jnp.arange(3.0))

    np.testing.assert_array_equal(result.states[0, 0], [2.0])
    np.testing.assert_array_equal(result.controls[0, 0], [0.0])
    # The first observation is 2, predicted mean is 0, gain is .45/.65.
    np.testing.assert_allclose(result.filtered_states_mean[0, 1], [2 * 0.45 / 0.65])
    np.testing.assert_allclose(result.controls[0, 1], [-0.45 / 0.65])
    np.testing.assert_array_equal(result.observations[0], result.states[0, 1:])


@pytest.mark.parametrize(
    "mismatch,match",
    [
        ("state_dim", "state_dim"),
        ("observation_dim", "observation_dim"),
        ("control_dim", "control_dim"),
        ("scalar", "sample shape"),
        ("continuous_time", "discrete-time"),
        ("alignment", "observation_control_alignment"),
        ("t0", "filter_dynamics.t0"),
    ],
)
def test_filter_dynamics_rejects_incompatible_models(mismatch, match):
    dynamics = _environment()
    if mismatch.endswith("_dim"):
        filter_dynamics = _environment(**{mismatch: 2})
    elif mismatch == "scalar":
        filter_dynamics = _environment(scalar=True)
    elif mismatch == "continuous_time":
        filter_dynamics = dsx.DynamicalModel(
            initial_condition=dynamics.initial_condition,
            state_evolution=dsx.DeterministicContinuousTimeStateEvolution(
                drift=lambda x, u, t: x
            ),
            observation_model=dynamics.observation_model,
            control_dim=1,
        )
    elif mismatch == "alignment":
        filter_dynamics = _environment()
        filter_dynamics = dsx.DynamicalModel(
            initial_condition=filter_dynamics.initial_condition,
            state_evolution=filter_dynamics.state_evolution,
            observation_model=filter_dynamics.observation_model,
            control_dim=1,
            observation_control_alignment="same_time",
        )
    else:
        filter_dynamics = eqx.tree_at(
            lambda m: m.t0, dynamics, 1.0, is_leaf=lambda x: x is None
        )

    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match=match):
        dsx.simulate(
            dynamics,
            filter_dynamics=filter_dynamics,
            control_policy=_policy,
            rng_key=jr.key(0),
            predict_times=jnp.arange(3.0),
        )


def test_filter_dynamics_requires_closed_loop():
    with pytest.raises(ValueError, match="filter_dynamics requires control_policy"):
        dsx.simulate(
            _environment(),
            filter_dynamics=_relaxed(_environment()),
            rng_key=jr.key(0),
            predict_times=jnp.arange(3.0),
        )


def test_filter_dynamics_supports_jit_and_traced_start_time():
    dynamics = _environment()

    @jax.jit
    def simulate(filter_t0):
        filter_dynamics = eqx.tree_at(
            lambda m: m.t0, _relaxed(dynamics), filter_t0, is_leaf=lambda x: x is None
        )
        return dsx.simulate(
            dynamics,
            filter_dynamics=filter_dynamics,
            control_policy=_policy,
            filter_config=EKFConfig(),
            rng_key=jr.key(0),
            predict_times=jnp.arange(3.0),
        ).states

    np.testing.assert_allclose(simulate(jnp.asarray(0.0))[0, :, 0], [2.0, 1.0, 0.5])
    with pytest.raises(Exception, match="filter_dynamics.t0 must match"):
        jax.block_until_ready(simulate(jnp.asarray(1.0)))
