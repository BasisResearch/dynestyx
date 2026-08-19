"""Tests for the pure-JAX dsx.simulate entry point."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace
from numpyro.infer import Predictive

import dynestyx as dsx


def _make_discrete_dynamics() -> dsx.DynamicalModel:
    return dsx.LTI_discrete(
        A=jnp.array([[0.7, 0.1], [0.0, 0.9]]),
        Q=0.05 * jnp.eye(2),
        H=jnp.array([[1.0, 0.0]]),
        R=jnp.array([[0.1**2]]),
    )


def _make_callable_discrete_dynamics() -> dsx.DynamicalModel:
    A = jnp.array([[0.7, 0.1], [0.0, 0.9]])
    Q = 0.05 * jnp.eye(2)
    H = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.1**2]])

    return dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2),
            covariance_matrix=0.2**2 * jnp.eye(2),
        ),
        state_evolution=lambda x, u, t_now, t_next: dist.MultivariateNormal(
            loc=A @ x,
            covariance_matrix=Q,
        ),
        observation_model=lambda x, u, t: dist.MultivariateNormal(
            loc=H @ x,
            covariance_matrix=R,
        ),
    )


def _make_ode_dynamics() -> dsx.DynamicalModel:
    return dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1),
            covariance_matrix=0.2**2 * jnp.eye(1),
        ),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.3 * x
        ),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.05**2]]),
        ),
    )


def _make_dirac_ode_dynamics() -> dsx.DynamicalModel:
    return dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1),
            covariance_matrix=0.2**2 * jnp.eye(1),
        ),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.3 * x
        ),
        observation_model=dsx.DiracIdentityObservation(),
    )


def _make_sde_dynamics() -> dsx.DynamicalModel:
    return dsx.DynamicalModel(
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(1),
            covariance_matrix=0.2**2 * jnp.eye(1),
        ),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.3 * x,
            diffusion=dsx.FullDiffusion(lambda x, u, t: 0.1 * jnp.eye(1)),
        ),
        observation_model=dsx.LinearGaussianObservation(
            H=jnp.array([[1.0]]),
            R=jnp.array([[0.05**2]]),
        ),
    )


def _make_controlled_deterministic_discrete_dynamics() -> dsx.DynamicalModel:
    return dsx.DynamicalModel(
        control_dim=1,
        initial_condition=dist.Delta(jnp.array([0.0])).to_event(1),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(x + u).to_event(1),
        observation_model=lambda x, u, t: dist.Delta(x).to_event(1),
    )


def test_simulate_discrete_returns_simulated_result():
    predict_times = jnp.arange(5.0)
    result = dsx.simulate(
        _make_discrete_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        n_simulations=3,
    )
    times = jnp.asarray(result.times)
    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)

    assert isinstance(result, dsx.SimulatedResult)
    assert result.x_0 is not None
    assert times.shape == (3, len(predict_times))
    assert result.x_0.shape == (3, 2)
    assert states.shape == (3, len(predict_times), 2)
    assert observations.shape == (3, len(predict_times), 1)
    assert jnp.allclose(times[0], predict_times)
    assert jnp.all(jnp.isfinite(states))
    assert jnp.all(jnp.isfinite(observations))


def test_simulate_vmaps_over_batched_controls():
    dynamics = _make_controlled_deterministic_discrete_dynamics()
    predict_times = jnp.arange(4.0)
    control_batch = jnp.array(
        [
            [[1.0], [1.0], [1.0], [1.0]],
            [[2.0], [2.0], [2.0], [2.0]],
        ]
    )

    result = jax.vmap(
        lambda ctrl_values: dsx.simulate(
            dynamics,
            rng_key=jr.PRNGKey(0),
            ctrl_times=predict_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            n_simulations=2,
        )
    )(control_batch)

    assert isinstance(result, dsx.SimulatedResult)
    assert result.states is not None
    assert result.states.shape == (2, 2, 4, 1)
    expected_states = jnp.array(
        [
            [[[0.0], [1.0], [2.0], [3.0]]],
            [[[0.0], [2.0], [4.0], [6.0]]],
        ]
    )
    assert jnp.array_equal(
        result.states, jnp.broadcast_to(expected_states, (2, 2, 4, 1))
    )


def test_simulate_callable_discrete_transition_auto_routes_to_discrete():
    predict_times = jnp.arange(5.0)
    result = dsx.simulate(
        _make_callable_discrete_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        n_simulations=2,
    )
    times = jnp.asarray(result.times)
    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)

    assert isinstance(result, dsx.SimulatedResult)
    assert times.shape == (2, len(predict_times))
    assert states.shape == (2, len(predict_times), 2)
    assert observations.shape == (2, len(predict_times), 1)
    assert jnp.allclose(times[0], predict_times)
    assert jnp.all(jnp.isfinite(states))
    assert jnp.all(jnp.isfinite(observations))


def test_condition_with_simulator_returns_deferred_simulated_result():
    predict_times = jnp.arange(5.0)

    with dsx.DiscreteTimeSimulator(n_simulations=2):
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            result = dsx.condition(
                "f",
                _make_discrete_dynamics(),
                predict_times=predict_times,
            )

    assert isinstance(result, dsx.SimulatedResult)
    assert callable(result._register_numpyro_sites)
    assert "f_x_0" not in tr
    assert "f_times" not in tr
    assert "f_states" not in tr
    assert "f_observations" not in tr


def test_sample_with_ode_simulator_allows_dirac_ode_models():
    predict_times = jnp.linspace(0.0, 1.0, 5)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with dsx.ODESimulator():
            result = dsx.sample(
                "f",
                _make_dirac_ode_dynamics(),
                predict_times=predict_times,
            )

    assert isinstance(result, dsx.SimulatedResult)
    assert "f_x_0" in tr
    assert "f_states" in tr
    assert "f_observations" in tr


def test_simulate_ode_accepts_structured_config():
    predict_times = jnp.linspace(0.0, 1.0, 6)
    result = dsx.simulate(
        _make_ode_dynamics(),
        rng_key=jr.PRNGKey(1),
        predict_times=predict_times,
        n_simulations=2,
        simulator_config=dsx.ODESimulatorConfig(dt0=0.05, max_steps=1_000),
    )
    times = jnp.asarray(result.times)
    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)

    assert isinstance(result, dsx.SimulatedResult)
    assert times.shape == (2, len(predict_times))
    assert states.shape == (2, len(predict_times), 1)
    assert observations.shape == (2, len(predict_times), 1)
    assert jnp.all(jnp.isfinite(states))


def test_simulate_sde_accepts_structured_config():
    predict_times = jnp.linspace(0.0, 0.5, 5)
    result = dsx.simulate(
        _make_sde_dynamics(),
        rng_key=jr.PRNGKey(2),
        predict_times=predict_times,
        n_simulations=2,
        simulator_config=dsx.SDESimulatorConfig(
            dt0=0.01,
            tol_vbt=0.005,
            max_steps=2_000,
            source="em_scan",
        ),
    )
    times = jnp.asarray(result.times)
    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)

    assert isinstance(result, dsx.SimulatedResult)
    assert times.shape == (2, len(predict_times))
    assert states.shape == (2, len(predict_times), 1)
    assert observations.shape == (2, len(predict_times), 1)
    assert jnp.all(jnp.isfinite(states))


def test_simulate_rejects_missing_time_grid():
    with pytest.raises(ValueError, match="predict_times must be provided"):
        dsx.simulate(_make_discrete_dynamics(), rng_key=jr.PRNGKey(0))


def test_simulate_allows_dirac_ode_models():
    predict_times = jnp.linspace(0.0, 1.0, 5)
    result = dsx.simulate(
        _make_dirac_ode_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
    )
    times = jnp.asarray(result.times)
    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)

    assert isinstance(result, dsx.SimulatedResult)
    assert times.shape == (1, len(predict_times))
    assert states.shape == (1, len(predict_times), 1)
    assert observations.shape == (1, len(predict_times), 1)
    assert jnp.allclose(observations, states)


def test_discrete_simulator_backend_aligns_ctrl_values_using_ctrl_times():
    predict_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    ctrl_times = jnp.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    ctrl_values = jnp.array([[999.0], [1.0], [10.0], [100.0], [1000.0]])

    result = dsx.DiscreteTimeSimulator().simulate(
        _make_controlled_deterministic_discrete_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    expected_states = jnp.array([[[0.0], [1.0], [11.0], [111.0]]])
    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)

    assert jnp.array_equal(states, expected_states)
    assert jnp.array_equal(observations, expected_states)


def test_discrete_simulator_backend_rejects_missing_ctrl_times():
    predict_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    ctrl_times = jnp.array([0.0, 1.0, 3.0])
    ctrl_values = jnp.array([[1.0], [10.0], [1000.0]])

    with pytest.raises(
        ValueError,
        match="ctrl_times must contain every discrete simulation time exactly",
    ):
        dsx.DiscreteTimeSimulator().simulate(
            _make_controlled_deterministic_discrete_dynamics(),
            rng_key=jr.PRNGKey(0),
            predict_times=predict_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )


@pytest.mark.parametrize("n_simulations", [1, 2])
@pytest.mark.parametrize(
    ("dynamics_factory", "predict_times", "simulator_config"),
    [
        pytest.param(
            _make_discrete_dynamics,
            jnp.arange(5.0),
            None,
            id="discrete-time",
        ),
        pytest.param(
            _make_ode_dynamics,
            jnp.linspace(0.0, 1.0, 6),
            dsx.ODESimulatorConfig(dt0=0.05, max_steps=1_000),
            id="ode",
        ),
        pytest.param(
            _make_sde_dynamics,
            jnp.linspace(0.0, 0.5, 5),
            dsx.SDESimulatorConfig(
                dt0=0.01,
                tol_vbt=0.005,
                max_steps=2_000,
                source="em_scan",
            ),
            id="sde",
        ),
    ],
)
def test_predictive_simulator_matches_standalone_simulate(
    n_simulations,
    dynamics_factory,
    predict_times,
    simulator_config,
):
    """The NumPyro and pure-JAX simulation entry points consume one key alike."""
    dynamics = dynamics_factory()
    rng_key = jr.PRNGKey(123)

    standalone = dsx.simulate(
        dynamics,
        rng_key=rng_key,
        predict_times=predict_times,
        n_simulations=n_simulations,
        simulator_config=simulator_config,
    )

    # Simulator.simulate accepts an already-allocated simulation key, whereas
    # dsx.simulate accepts the same root key passed to Predictive.
    _, simulation_key = jr.split(rng_key)
    direct_simulator = dsx.Simulator(
        n_simulations=n_simulations,
        simulator_config=simulator_config,
    ).simulate(
        dynamics,
        rng_key=simulation_key,
        predict_times=predict_times,
    )

    def model():
        return dsx.sample("f", dynamics, predict_times=predict_times)

    with dsx.Simulator(
        n_simulations=n_simulations,
        simulator_config=simulator_config,
    ):
        predictive = Predictive(
            model,
            num_samples=1,
            exclude_deterministic=False,
        )(rng_key)

    mismatched_fields = []
    for field in ("x_0", "times", "states", "observations"):
        standalone_value = getattr(standalone, field)
        direct_simulator_value = getattr(direct_simulator, field)
        predictive_value = predictive[f"f_{field}"][0]
        if not (
            jnp.array_equal(predictive_value, standalone_value)
            and jnp.array_equal(direct_simulator_value, standalone_value)
        ):
            mismatched_fields.append(field)

    assert not mismatched_fields, (
        "Simulator + Predictive, dsx.simulate, and pre-split Simulator.simulate "
        f"produced different values for {mismatched_fields}"
    )


# ---------------------------------------------------------------------------
# observation_control_alignment="previous_transition" (#312)
# ---------------------------------------------------------------------------


def _make_previous_transition_dynamics() -> dsx.DynamicalModel:
    """Deterministic 1-D discrete model whose observation reveals both the
    state and (scaled) control it was conditioned on, so tests can check
    exactly which control an observation used."""

    def _state_evolution(x, u, t_now, t_next):
        del t_now, t_next
        u = jnp.zeros_like(x) if u is None else u
        return dist.Delta(x + u).to_event(1)

    def _observation_model(x, u, t):
        del t
        u = jnp.zeros_like(x) if u is None else u
        return dist.Delta(x + 100.0 * u).to_event(1)

    return dsx.DynamicalModel(
        control_dim=1,
        initial_condition=dist.Delta(jnp.array([0.0])).to_event(1),
        state_evolution=_state_evolution,
        observation_model=_observation_model,
        observation_control_alignment="previous_transition",
    )


def test_discrete_simulator_previous_transition_aligns_ctrl_values_shorter_by_one():
    predict_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    ctrl_times = predict_times[:-1]
    ctrl_values = jnp.array([[1.0], [2.0], [3.0]])

    result = dsx.DiscreteTimeSimulator().simulate(
        _make_previous_transition_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)
    times = jnp.asarray(result.times)

    assert result.x_0 is None
    assert times.shape == (1, 3)
    assert jnp.allclose(times[0], predict_times[1:])
    assert states.shape == (1, 3, 1)
    assert observations.shape == (1, 3, 1)
    # x_1=1, x_2=3, x_3=6 (cumulative sum of controls, x_0=0)
    expected_states = jnp.array([[[1.0], [3.0], [6.0]]])
    assert jnp.array_equal(states, expected_states)
    # y_{k+1} = x_{k+1} + 100 * u_k
    expected_observations = expected_states + 100.0 * jnp.array([[[1.0], [2.0], [3.0]]])
    assert jnp.array_equal(observations, expected_observations)


def test_discrete_simulator_previous_transition_rejects_ctrl_times_matching_full_predict_times():
    """dsx.simulate's _validate_controls gate requires an exact-length match
    against predict_times[:-1] for previous_transition; DiscreteTimeSimulator's
    own _align_ctrl_values_to_times permits a superset ctrl_times, so this must
    go through the public dsx.simulate entry point to observe the rejection."""
    predict_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    ctrl_values = jnp.array([[1.0], [2.0], [3.0], [4.0]])

    with pytest.raises(Exception):
        dsx.simulate(
            _make_previous_transition_dynamics(),
            rng_key=jr.PRNGKey(0),
            predict_times=predict_times,
            ctrl_times=predict_times,
            ctrl_values=ctrl_values,
        )


def test_dsx_simulate_previous_transition_end_to_end():
    """Exercises dsx.simulate -> api.py -> utils.py::_validate_controls threading."""
    predict_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    ctrl_times = predict_times[:-1]
    ctrl_values = jnp.array([[1.0], [2.0], [3.0]])

    result = dsx.simulate(
        _make_previous_transition_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    assert result.x_0 is None
    assert jnp.asarray(result.times).shape == (1, 3)
    assert jnp.asarray(result.states).shape == (1, 3, 1)
    assert jnp.asarray(result.observations).shape == (1, 3, 1)


def test_discrete_simulator_previous_transition_zero_length_predict_times_edge_case():
    result = dsx.simulate(
        _make_previous_transition_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=jnp.arange(1.0),
    )

    assert result.x_0 is None
    assert jnp.asarray(result.times).shape == (1, 0)
    assert jnp.asarray(result.states).shape == (1, 0, 1)
    assert jnp.asarray(result.observations).shape == (1, 0, 1)


def test_discrete_simulator_previous_transition_rejects_obs_times():
    predict_times = jnp.array([0.0, 1.0, 2.0])

    with pytest.raises(
        ValueError,
        match="observation_control_alignment='previous_transition' does not support "
        "obs_times",
    ):
        dsx.condition(
            "f",
            _make_previous_transition_dynamics(),
            obs_times=predict_times,
            obs_values=jnp.zeros((3, 1)),
            predict_times=predict_times,
        )


def test_previous_transition_observation_uses_previous_step_control_not_same_index():
    """y_{k+1} must use u_k (the control that produced x_{k+1}), not u_{k+1}."""
    predict_times = jnp.array([0.0, 1.0, 2.0, 3.0])
    ctrl_times = predict_times[:-1]
    ctrl_values = jnp.array([[1.0], [2.0], [3.0]])

    result = dsx.DiscreteTimeSimulator().simulate(
        _make_previous_transition_dynamics(),
        rng_key=jr.PRNGKey(0),
        predict_times=predict_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    states = jnp.asarray(result.states)
    observations = jnp.asarray(result.observations)
    # observation_model reveals x + 100*u, so subtracting the realized state
    # recovers exactly which (scaled) control each observation used.
    revealed_control = (observations - states) / 100.0
    assert jnp.array_equal(revealed_control[0], ctrl_values)
