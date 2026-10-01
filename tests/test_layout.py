"""Standalone layouts and explicit simulation-result conversions."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed

import dynestyx as dsx


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


def test_layout_collection_from_example_and_with_examples():
    state_example = {"position": jnp.zeros(2)}
    observation_example = jnp.zeros((2, 3))
    control_example = (jnp.zeros(1), jnp.zeros(2))

    state_only = dsx.LayoutCollection.from_example(state=state_example)
    assert state_only.state is not None
    assert state_only.state.dim == 2
    assert state_only.control is None
    assert state_only.observation is None

    extended = state_only.with_examples(
        control=control_example, observation=observation_example
    )
    assert extended is not state_only
    assert extended.state is state_only.state
    assert extended.control is not None
    assert extended.control.dim == 3
    assert extended.observation is not None
    assert extended.observation.dim == 6
    assert state_only.control is None
    assert state_only.observation is None


def test_layout_collection_is_static_equinox_module():
    layout = dsx.LayoutCollection.from_example(state=(jnp.zeros(2), jnp.zeros(1)))
    assert isinstance(layout, eqx.Module)
    assert isinstance(layout.state, eqx.Module)
    assert jax.tree_util.tree_leaves(layout) == []

    @jax.jit
    def unflatten(collection, value):
        return collection.state.unflatten(value)

    restored = unflatten(layout, jnp.arange(3.0))
    assert jnp.array_equal(restored[0], jnp.array([0.0, 1.0]))
    assert jnp.array_equal(restored[1], jnp.array([2.0]))


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
    initial_state = {
        "latent": jnp.array([0.5, -0.25]),
        "observed": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
    }
    initial_flat = jnp.array([0.5, -0.25, 1.0, 2.0, 3.0, 4.0])
    layout = dsx.LayoutCollection.from_example(
        state=initial_state, observation=jnp.zeros((2, 2))
    )
    state_layout = layout.state
    observation_layout = layout.observation
    assert state_layout is not None
    assert observation_layout is not None
    assert jnp.array_equal(state_layout.flatten(initial_state), initial_flat)

    A = jnp.array([[0.1, 0.2], [-0.3, 0.4]])
    alpha = 0.8

    def structured_transition(x, u, t_now, t_next):
        state = state_layout.unflatten(x)
        next_state = {
            "latent": alpha * state["latent"],
            "observed": state["observed"]
            + (t_next - t_now) * (A @ state["latent"])[:, None],
        }
        return dist.Normal(state_layout.flatten(next_state), 0.1).to_event(1)

    def flat_transition(x, u, t_now, t_next):
        latent = x[:2]
        observed = x[2:].reshape(2, 2)
        next_observed = observed + (t_next - t_now) * (A @ latent)[:, None]
        loc = jnp.concatenate((alpha * latent, next_observed.reshape(-1)))
        return dist.Normal(loc, 0.1).to_event(1)

    def structured_observation(x, u, t):
        state = state_layout.unflatten(x)
        observation = jnp.logaddexp(0.0, state["observed"])
        return dist.Normal(observation_layout.flatten(observation), 0.2).to_event(1)

    def flat_observation(x, u, t):
        observation = jnp.logaddexp(0.0, x[2:].reshape(2, 2))
        return dist.Normal(observation.reshape(-1), 0.2).to_event(1)

    initial_condition = dist.Normal(initial_flat, 0.05).to_event(1)
    structured_dynamics = dsx.DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=structured_transition,
        observation_model=structured_observation,
    )
    flat_dynamics = dsx.DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=flat_transition,
        observation_model=flat_observation,
    )
    times = jnp.array([0.0, 1.0, 2.5, 3.0])
    key = jr.key(0)
    structured_model_result = dsx.simulate(
        structured_dynamics, rng_key=key, predict_times=times, n_simulations=3
    )
    flat_model_result = dsx.simulate(
        flat_dynamics, rng_key=key, predict_times=times, n_simulations=3
    )

    assert structured_model_result.x_0 is not None
    assert structured_model_result.states is not None
    assert structured_model_result.observations is not None
    assert flat_model_result.x_0 is not None
    assert flat_model_result.states is not None
    assert flat_model_result.observations is not None
    assert structured_model_result.states.shape == (3, 4, 6)
    assert structured_model_result.observations.shape == (3, 4, 4)
    assert jnp.allclose(structured_model_result.x_0, flat_model_result.x_0)
    assert jnp.allclose(structured_model_result.states, flat_model_result.states)
    assert jnp.allclose(
        structured_model_result.observations, flat_model_result.observations
    )

    result = structured_model_result.unflatten(layout)
    assert result.states is not None
    assert result.observations is not None
    assert result.states["latent"].shape == (3, 4, 2)
    assert result.states["observed"].shape == (3, 4, 2, 2)
    assert result.observations.shape == (3, 4, 2, 2)
    assert jnp.allclose(result.states["latent"], flat_model_result.states[..., :2])
    assert jnp.allclose(
        result.states["observed"], flat_model_result.states[..., 2:].reshape(3, 4, 2, 2)
    )
    assert jnp.allclose(
        result.observations, flat_model_result.observations.reshape(3, 4, 2, 2)
    )


@pytest.mark.parametrize("selected", ["state", "control", "observation"])
def test_result_converts_only_selected_sublayout(selected):
    all_layouts = _layouts()
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
