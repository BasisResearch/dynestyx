"""Standalone layouts and explicit simulation-result conversions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest

import dynestyx as dsx


def _layouts():
    return dsx.Layouts(
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


def test_layout_round_trip_with_scalar_leaf_and_batch_axes():
    example = {"a": jnp.zeros((2, 2)), "b": (jnp.zeros(()), jnp.zeros(1))}
    layout = dsx.Layout.from_example(example)
    value = {
        "a": jnp.arange(24.0).reshape(2, 3, 2, 2),
        "b": (jnp.ones((2, 3)), jnp.ones((2, 3, 1))),
    }

    flat = jax.jit(layout.flatten)(value)
    restored = jax.jit(layout.unflatten)(flat)

    assert layout.dim == 6
    assert flat.shape == (2, 3, 6)
    assert all(jax.tree.leaves(jax.tree.map(jnp.array_equal, restored, value)))


def test_layout_rejects_mismatched_values():
    layout = dsx.Layout.from_example((jnp.zeros(2), jnp.zeros(1)))
    with pytest.raises(ValueError, match="structure"):
        layout.flatten({"a": jnp.zeros(2), "b": jnp.zeros(1)})
    with pytest.raises(ValueError, match="trailing shape"):
        layout.flatten((jnp.zeros(3), jnp.zeros(1)))
    with pytest.raises(ValueError, match="leading batch axes"):
        layout.flatten((jnp.zeros((2, 2)), jnp.zeros((3, 1))))
    with pytest.raises(ValueError, match="trailing flat axis"):
        layout.unflatten(jnp.zeros(2))


def test_layout_rejects_empty_and_mixed_dtype_trees():
    with pytest.raises(ValueError, match="nonempty"):
        dsx.Layout.from_example({})
    with pytest.raises(TypeError, match="same numeric dtype"):
        dsx.Layout.from_example((jnp.zeros(1), jnp.zeros(1, dtype=jnp.int32)))
    with pytest.raises(ValueError, match="zero-sized"):
        dsx.Layout.from_example(jnp.zeros(0))


def test_dynamics_can_close_over_layout_without_a_new_model_contract():
    state = dsx.Layout.from_example({"position": jnp.zeros(2)})
    layout = dsx.Layouts(state=state)

    def transition(x, u, t_now, t_next):
        position = state.unflatten(x)["position"]
        next_position = position + 1
        return dist.Delta(state.flatten({"position": next_position})).to_event(1)

    dynamics = dsx.DynamicalModel(
        initial_condition=dist.Delta(jnp.zeros(2)).to_event(1),
        state_evolution=transition,
        observation_model=lambda x, u, t: dist.Delta(x).to_event(1),
    )
    flat = dsx.simulate(dynamics, rng_key=jr.key(0), predict_times=jnp.arange(3.0))
    structured = flat.unflatten(layout)

    assert flat.states is not None
    assert structured.states is not None
    assert flat.states.shape == (1, 3, 2)
    assert jnp.array_equal(
        structured.states["position"][0, :, 0], jnp.array([0.0, 1.0, 2.0])
    )


@pytest.mark.parametrize("selected", ["state", "control", "observation"])
def test_result_converts_only_selected_sublayout(selected):
    all_layouts = _layouts()
    layout = dsx.Layouts(**{selected: getattr(all_layouts, selected)})
    times = jnp.arange(3.0)
    flat = dsx.simulate(
        _controlled_model(),
        rng_key=jr.key(0),
        predict_times=times,
        ctrl_times=times,
        ctrl_values=jnp.ones((3, 2)),
        n_simulations=2,
    )
    structured = flat.unflatten(layout)
    restored = structured.flatten(layout)

    assert flat.x_0 is not None
    assert flat.states is not None
    assert flat.observations is not None
    assert flat.controls is not None
    assert structured.x_0 is not None
    assert structured.states is not None
    assert structured.observations is not None
    assert structured.controls is not None
    assert flat.states.shape == (2, 3, 4)
    assert flat.controls.shape == (2, 3, 2)
    assert flat.observations.shape == (2, 3, 2)
    assert structured.times is flat.times
    assert structured.obs_times is flat.obs_times
    assert structured.ctrl_times is flat.ctrl_times
    if selected == "state":
        assert structured.x_0["position"].shape == (2, 2)
        assert structured.states["velocity"].shape == (2, 3, 2)
        assert structured.controls.shape == flat.controls.shape
    elif selected == "control":
        assert structured.controls[0].shape == (2, 3, 1)
        assert structured.states.shape == flat.states.shape
    else:
        assert structured.observations["measured"].shape == (2, 3, 2)
        assert structured.states.shape == flat.states.shape

    for name in ("x_0", "states", "observations", "controls"):
        original = getattr(flat, name)
        round_trip = getattr(restored, name)
        assert original is not None
        assert round_trip is not None
        assert jnp.array_equal(round_trip, original)


def test_result_all_layouts_round_trip_under_jit():
    layout = _layouts()
    times = jnp.arange(3.0)
    controls = jnp.ones((3, 2))

    @jax.jit
    def run(key):
        flat = dsx.simulate(
            _controlled_model(),
            rng_key=key,
            predict_times=times,
            ctrl_times=times,
            ctrl_values=controls,
        )
        structured = flat.unflatten(layout)
        return structured, structured.flatten(layout)

    structured, flat = run(jr.key(0))
    assert structured.states is not None
    assert structured.controls is not None
    assert structured.observations is not None
    assert flat.states is not None
    assert flat.controls is not None
    assert structured.states["position"].shape == (1, 3, 2)
    assert structured.controls[1].shape == (1, 3, 1)
    assert structured.observations["measured"].shape == (1, 3, 2)
    assert flat.states.shape == (1, 3, 4)
    assert flat.controls.shape == (1, 3, 2)


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
    layout = dsx.Layouts(state=dsx.Layout.from_example({"value": jnp.array(0.0)}))
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
