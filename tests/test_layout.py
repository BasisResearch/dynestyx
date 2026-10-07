"""Standalone layouts and explicit simulation-result conversions."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from jaxtyping import TypeCheckError
from numpyro.handlers import seed

import dynestyx as dsx
from dynestyx.control import (
    ControlledSimulatedResult,
    StructuredControlledSimulatedResult,
    filter_state_mean,
)
from dynestyx.inference.configs.filter import EKFConfig


def _layouts():
    return dsx.LayoutCollection(
        state=dsx.Layout.from_example(
            {"position": jnp.zeros(2), "velocity": jnp.zeros(2)}
        ),
        control=dsx.Layout.from_example((jnp.zeros(1), jnp.zeros(1))),
        observation=dsx.Layout.from_example({"measured": jnp.zeros(2)}),
    )


def _controlled_model():
    return dsx.DynamicalModel(
        control_dim=2,
        initial_condition=dist.Delta(jnp.zeros(4)).to_event(1),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(
            x + jnp.concatenate((u, jnp.zeros(2)))
        ).to_event(1),
        observation_model=lambda x, u, t: dist.Delta(x[:2]).to_event(1),
    )


def test_layout_from_example():
    state_example = {"position": jnp.zeros(2)}
    observation_example = jnp.zeros((2, 3))
    control_example = (jnp.zeros(1), jnp.zeros(2))

    state_only = dsx.LayoutCollection.from_example(state=state_example)
    assert state_only.state is not None
    assert state_only.state.dim == 2
    assert state_only.control is None
    assert state_only.observation is None

    full = dsx.LayoutCollection.from_example(
        state=state_example,
        control=control_example,
        observation=observation_example,
    )
    assert full.state is not None
    assert full.state.dim == 2
    assert full.control is not None
    assert full.control.dim == 3
    assert full.observation is not None
    assert full.observation.dim == 6

    # Layouts are static Equinox modules, so they hold no array leaves.
    assert isinstance(full, eqx.Module)
    assert isinstance(full.state, eqx.Module)
    assert jax.tree_util.tree_leaves(full) == []


def test_layout_round_trip_under_jit():
    example = {"a": jnp.zeros((2, 2)), "b": (jnp.zeros(()), jnp.zeros(1))}
    layout = dsx.LayoutCollection.from_example(state=example)
    assert layout.state is not None
    value = {
        "a": jnp.arange(24.0).reshape(2, 3, 2, 2),
        "b": (jnp.ones((2, 3)), jnp.ones((2, 3, 1))),
    }

    @jax.jit
    def round_trip(layout, value):
        flat = layout.state.flatten(value)
        return flat, layout.state.unflatten(flat)

    flat, restored = round_trip(layout, value)

    assert layout.state.dim == 6
    assert flat.shape == (2, 3, 6)
    assert all(jax.tree.leaves(jax.tree.map(jnp.array_equal, restored, value)))


def test_plated_dynamics_match_explicit_flat_model():
    layout = dsx.LayoutCollection.from_example(
        state={"left": jnp.zeros(1), "right": jnp.zeros(1)},
        control={"drive": jnp.zeros(1)},
        observation=(jnp.zeros(1), jnp.zeros(1)),
    )
    state_layout = layout.state
    control_layout = layout.control
    observation_layout = layout.observation
    assert state_layout is not None
    assert control_layout is not None
    assert observation_layout is not None

    def structured_transition(x, u, t_now, t_next):
        state = state_layout.unflatten(x)
        drive = control_layout.unflatten(u)["drive"]
        next_state = {
            "left": 0.5 * state["left"] + jnp.sin(drive),
            "right": 0.5 * state["right"] - jnp.sin(drive),
        }
        return dist.Normal(state_layout.flatten(next_state), 0.1).to_event(1)

    def flat_transition(x, u, t_now, t_next):
        loc = 0.5 * x + jnp.concatenate((jnp.sin(u), -jnp.sin(u)))
        return dist.Normal(loc, 0.1).to_event(1)

    def structured_observation(x, u, t):
        state = state_layout.unflatten(x)
        loc = observation_layout.flatten(
            (jnp.logaddexp(0.0, state["left"]), jnp.sin(state["right"]))
        )
        return dist.Normal(loc, 0.2).to_event(1)

    def flat_observation(x, u, t):
        loc = jnp.concatenate((jnp.logaddexp(0.0, x[:1]), jnp.sin(x[1:])))
        return dist.Normal(loc, 0.2).to_event(1)

    initial = jnp.array([[0.0, 10.0], [5.0, 20.0]])
    controls = jnp.array([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    times = jnp.arange(3.0)

    def simulate(transition, observation):
        with dsx.DiscreteTimeSimulator(n_simulations=2), seed(rng_seed=jr.key(0)):
            with dsx.plate("members", 2):
                dynamics = dsx.DynamicalModel(
                    control_dim=1,
                    initial_condition=dist.Normal(initial, 0.05).to_event(1),
                    state_evolution=transition,
                    observation_model=observation,
                )
                return dsx.sample(
                    "f",
                    dynamics,
                    predict_times=times,
                    ctrl_times=times,
                    ctrl_values=controls,
                )

    flat = simulate(structured_transition, structured_observation)
    explicit_flat = simulate(flat_transition, flat_observation)
    structured = flat.unflatten(layout)
    restored = structured.flatten(layout)
    assert flat.x_0 is not None
    assert flat.states is not None
    assert flat.controls is not None
    assert flat.observations is not None
    assert explicit_flat.x_0 is not None
    assert explicit_flat.states is not None
    assert explicit_flat.controls is not None
    assert explicit_flat.observations is not None
    assert structured.states is not None
    assert structured.controls is not None
    assert structured.observations is not None
    assert flat.states.shape == (2, 2, 3, 2)
    assert structured.states["left"].shape == (2, 2, 3, 1)
    assert structured.controls["drive"].shape == (2, 2, 3, 1)
    assert structured.observations[0].shape == (2, 2, 3, 1)
    assert jnp.array_equal(structured.controls["drive"][:, 0], controls)
    for name in ("x_0", "states", "controls", "observations"):
        assert jnp.allclose(getattr(flat, name), getattr(explicit_flat, name))
        assert jnp.array_equal(getattr(restored, name), getattr(flat, name))


@pytest.mark.parametrize("selected", ["state", "control", "observation", "all"])
def test_result_round_trip_under_jit(selected):
    all_layouts = _layouts()
    if selected == "all":
        layout = all_layouts
    else:
        layout = dsx.LayoutCollection(**{selected: getattr(all_layouts, selected)})
    times = jnp.arange(3.0)
    flat = dsx.simulate(
        _controlled_model(),
        rng_key=jr.key(0),
        predict_times=times,
        ctrl_times=times,
        ctrl_values=jnp.ones((3, 2)),
        n_simulations=2,
    )

    @jax.jit
    def round_trip(flat):
        structured = flat.unflatten(layout)
        return structured, structured.flatten(layout)

    structured, restored = round_trip(flat)

    assert isinstance(structured, dsx.StructuredSimulatedResult)
    assert not isinstance(structured, dsx.SimulatedResult)
    assert type(restored) is dsx.SimulatedResult

    # The original result is unchanged.
    assert flat.x_0 is not None
    assert flat.states is not None
    assert flat.controls is not None
    assert flat.observations is not None
    assert flat.x_0.shape == (2, 4)
    assert flat.states.shape == (2, 3, 4)
    assert flat.controls.shape == (2, 3, 2)
    assert flat.observations.shape == (2, 3, 2)

    # Only the selected fields are converted; time fields never are.
    expected_shapes = {
        "x_0": ("state", lambda value: value["position"].shape, (2, 2)),
        "states": ("state", lambda value: value["velocity"].shape, (2, 3, 2)),
        "controls": ("control", lambda value: value[0].shape, (2, 3, 1)),
        "observations": (
            "observation",
            lambda value: value["measured"].shape,
            (2, 3, 2),
        ),
    }
    for name, (sublayout_name, leaf_shape, shape) in expected_shapes.items():
        value = getattr(structured, name)
        if selected in (sublayout_name, "all"):
            assert leaf_shape(value) == shape
        else:
            assert jnp.array_equal(value, getattr(flat, name))
    for name in ("times", "obs_times", "ctrl_times"):
        assert jnp.array_equal(getattr(structured, name), getattr(flat, name))

    for name in ("x_0", "states", "observations", "controls"):
        original = getattr(flat, name)
        round_trip = getattr(restored, name)
        assert round_trip is not None
        assert jnp.array_equal(round_trip, original)


@pytest.mark.parametrize("scalar_initial_event", [True, False])
@pytest.mark.parametrize("batched", [True, False])
def test_scalar_layout_preserves_initial_state_shape(scalar_initial_event, batched):
    x0 = jnp.array(0.0) if scalar_initial_event else jnp.array([0.0])
    event_dim = 0 if scalar_initial_event else 1
    dynamics = dsx.DynamicalModel(
        initial_condition=dist.Delta(x0, event_dim=event_dim),
        state_evolution=lambda x, u, t_now, t_next: dist.Delta(
            x + 1, event_dim=event_dim
        ),
        observation_model=lambda x, u, t: dist.Delta(x, event_dim=event_dim),
    )
    layout = dsx.LayoutCollection(
        state=dsx.Layout.from_example({"value": jnp.array(0.0)})
    )
    times = jnp.arange(3.0)
    if batched:
        flat = jax.vmap(
            lambda key: dsx.simulate(dynamics, rng_key=key, predict_times=times)
        )(jr.split(jr.key(0), 2))
    else:
        flat = dsx.simulate(dynamics, rng_key=jr.key(0), predict_times=times)

    structured = flat.unflatten(layout)
    restored = structured.flatten(layout)

    assert flat.x_0 is not None
    assert flat.states is not None
    assert structured.x_0 is not None
    assert structured.states is not None
    assert restored.x_0 is not None
    assert restored.states is not None
    leading = (2,) if batched else ()
    assert flat.x_0.shape == (
        (*leading, 1) if scalar_initial_event else (*leading, 1, 1)
    )
    assert structured.x_0["value"].shape == (*leading, 1)
    assert structured.states["value"].shape == (*leading, 1, 3)
    assert restored.x_0.shape == flat.x_0.shape
    assert jnp.array_equal(restored.x_0, flat.x_0)
    assert jnp.array_equal(restored.states, flat.states)


def test_predicted_fields_round_trip_and_callback_is_retained():
    layout = _layouts()
    callback = lambda name: None
    flat = dsx.SimulatedResult(
        predicted_times=jnp.zeros((2, 3)),
        predicted_states=jnp.ones((2, 3, 4)),
        predicted_observations=jnp.ones((2, 3, 2)),
        _register_numpyro_sites=callback,
    )

    structured = flat.unflatten(layout)
    restored = structured.flatten(layout)

    assert structured.predicted_states is not None
    assert structured.predicted_observations is not None
    assert restored.predicted_states is not None
    assert restored.predicted_observations is not None
    assert flat.predicted_states is not None
    assert flat.predicted_observations is not None
    assert structured.predicted_states["position"].shape == (2, 3, 2)
    assert structured.predicted_observations["measured"].shape == (2, 3, 2)
    assert structured.predicted_times is flat.predicted_times
    assert structured._register_numpyro_sites is callback
    assert restored._register_numpyro_sites is callback
    assert jnp.array_equal(restored.predicted_states, flat.predicted_states)
    assert jnp.array_equal(restored.predicted_observations, flat.predicted_observations)


def test_controlled_result_round_trip_under_jit():
    layout = _layouts()
    dynamics = dsx.LTI_discrete(
        A=0.9 * jnp.eye(4),
        Q=0.05 * jnp.eye(4),
        H=jnp.eye(2, 4),
        R=0.1 * jnp.eye(2),
        B=jnp.eye(4, 2),
        observation_control_alignment="previous_transition",
    )

    def policy(x_hat, t_now, t_next, s):
        return -0.5 * filter_state_mean(x_hat)[:2], s + 1.0

    flat = dsx.simulate(
        dynamics,
        rng_key=jr.key(0),
        predict_times=jnp.arange(4.0),
        control_policy=policy,
        initial_policy_state=jnp.array(0.0),
        filter_config=EKFConfig(record_filtered_states_mean=True),
    )

    @jax.jit
    def round_trip(flat):
        structured = flat.unflatten(layout)
        return structured, structured.flatten(layout)

    structured, restored = round_trip(flat)

    assert isinstance(flat, ControlledSimulatedResult)
    assert isinstance(structured, StructuredControlledSimulatedResult)
    assert type(restored) is ControlledSimulatedResult
    assert flat.filtered_states_mean is not None
    assert flat.policy_states is not None
    assert structured.filtered_states_mean is not None
    assert structured.controls is not None
    assert structured.filtered_states_mean["velocity"].shape == (
        *flat.filtered_states_mean.shape[:-1],
        2,
    )
    assert structured.controls[1].shape == (1, 3, 1)
    assert jnp.array_equal(structured.policy_states, flat.policy_states)
    for name in ("x_0", "states", "observations", "controls", "filtered_states_mean"):
        original = getattr(flat, name)
        round_trip = getattr(restored, name)
        # The original result is unchanged.
        assert eqx.is_array(original)
        assert round_trip is not None
        assert jnp.array_equal(round_trip, original)


@pytest.mark.parametrize(
    ("flat_cls", "structured_cls"),
    [
        (dsx.SimulatedResult, dsx.StructuredSimulatedResult),
        (ControlledSimulatedResult, StructuredControlledSimulatedResult),
    ],
)
def test_structured_results_mirror_flat_fields(flat_cls, structured_cls):
    flat = {field.name for field in dataclasses.fields(flat_cls)}
    structured = {field.name for field in dataclasses.fields(structured_cls)}
    assert structured - {"_scalar_x_0"} == flat


def test_invalid_inputs_raise():
    pair = dsx.Layout.from_example((jnp.zeros(2), jnp.zeros(1)))
    with pytest.raises(ValueError, match="structure"):
        pair.flatten({"a": jnp.zeros(2), "b": jnp.zeros(1)})
    with pytest.raises(ValueError, match="trailing shape"):
        pair.flatten((jnp.zeros(3), jnp.zeros(1)))
    with pytest.raises(ValueError, match="leading batch axes"):
        pair.flatten((jnp.zeros((2, 2)), jnp.zeros((3, 1))))
    with pytest.raises(ValueError, match="trailing flat axis"):
        pair.unflatten(jnp.zeros(2))
    with pytest.raises(ValueError, match="nonempty"):
        dsx.Layout.from_example({})
    with pytest.raises(TypeError, match="same numeric dtype"):
        dsx.Layout.from_example((jnp.zeros(1), jnp.zeros(1, dtype=jnp.int32)))
    with pytest.raises(ValueError, match="zero-sized"):
        dsx.Layout.from_example(jnp.zeros(0))

    layout = _layouts()
    assert layout.state is not None
    flat = dsx.SimulatedResult(
        times=jnp.zeros((1, 3)),
        states=jnp.ones((1, 3, 4)),
        observations=jnp.ones((1, 3, 2)),
    )
    with pytest.raises(TypeError, match="LayoutCollection"):
        flat.unflatten(layout.state)  # type: ignore[arg-type]
    structured = flat.unflatten(layout)
    with pytest.raises(ValueError, match="states is structured"):
        structured.flatten(dsx.LayoutCollection(observation=layout.observation))

    # The state fields must share one pytree structure.
    with pytest.raises(TypeCheckError):
        dsx.StructuredSimulatedResult(
            x_0={"position": jnp.zeros((1, 2))},
            states={"position": jnp.zeros((1, 3, 2)), "velocity": jnp.zeros((1, 3, 2))},
        )
