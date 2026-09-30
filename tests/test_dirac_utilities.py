"""Dirac model helpers for discrete-time simulation."""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist

import dynestyx as dsx


def test_structured_dirac_model_simulates_in_flat_coordinates():
    layout = dsx.Layouts(
        state=dsx.Layout.from_example(
            {"position": jnp.zeros(2), "velocity": jnp.zeros(2)}
        ),
        control=dsx.Layout.from_example((jnp.zeros(1), jnp.zeros(1))),
        observation=dsx.Layout.from_example({"measured": jnp.zeros(2)}),
    )

    def transition(x, u, t_now, t_next):
        return {
            "position": x["position"] + x["velocity"] + u[0],
            "velocity": x["velocity"] + u[1],
        }

    def observe(x, u, t):
        return {"measured": x["position"] + u[0]}

    dynamics = dsx.DynamicalModel(
        control_dim=layout.control.dim,
        initial_condition=dsx.DiracInitialCondition(
            {"position": jnp.array([1.0, 2.0]), "velocity": jnp.zeros(2)},
            layout=layout,
        ),
        state_evolution=dsx.DiracStateEvolution(transition, layout=layout),
        observation_model=dsx.DiracObservation(observe, layout=layout),
    )
    times = jnp.arange(3.0)
    result = jax.jit(
        lambda: dsx.simulate(
            dynamics,
            rng_key=jr.key(0),
            predict_times=times,
            ctrl_times=times,
            ctrl_values=jnp.tile(jnp.array([1.0, 2.0]), (3, 1)),
        )
    )()

    assert result.states.shape == (1, 3, 4)
    assert result.observations.shape == (1, 3, 2)
    structured = result.unflatten(layout)
    assert jnp.array_equal(
        structured.states["position"][0],
        jnp.array([[1.0, 2.0], [2.0, 3.0], [5.0, 6.0]]),
    )
    assert jnp.array_equal(
        structured.observations["measured"][0],
        jnp.array([[2.0, 3.0], [3.0, 4.0], [6.0, 7.0]]),
    )


def test_dirac_helpers_use_only_supplied_sublayouts():
    state = dsx.Layout.from_example({"value": jnp.zeros(2)})
    control = dsx.Layout.from_example((jnp.zeros(1), jnp.zeros(1)))
    observation = dsx.Layout.from_example({"value": jnp.zeros(2)})
    x = jnp.array([1.0, 2.0])
    u = jnp.array([3.0, 4.0])

    state_only = dsx.DiracStateEvolution(
        lambda x, u, *_: {"value": x["value"] + u[0]},
        layout=dsx.Layouts(state=state),
    )
    control_only = dsx.DiracStateEvolution(
        lambda x, u, *_: x + u[0],
        layout=dsx.Layouts(control=control),
    )
    observation_only = dsx.DiracObservation(
        lambda x, u, t: {"value": x + u[0]},
        layout=dsx.Layouts(observation=observation),
    )

    assert jnp.array_equal(state_only.mean(x, u, 0, 1), jnp.array([4.0, 5.0]))
    assert jnp.array_equal(control_only.mean(x, u, 0, 1), jnp.array([4.0, 5.0]))
    assert jnp.array_equal(observation_only.mean(x, u, 0), jnp.array([4.0, 5.0]))
    assert jnp.array_equal(
        dsx.DiracStateEvolution(lambda x, u, *_: x + 1).mean(x, None, 0, 1),
        jnp.array([2.0, 3.0]),
    )


def test_scalar_dirac_model_and_identity_observation():
    observation = dsx.DiracIdentityObservation()
    assert isinstance(observation, dsx.DiracObservation)

    dynamics = dsx.DynamicalModel(
        initial_condition=dsx.DiracInitialCondition(2.0),
        state_evolution=dsx.DiracStateEvolution(lambda x, u, *_: x + 1),
        observation_model=observation,
    )
    result = dsx.simulate(dynamics, rng_key=jr.key(0), predict_times=jnp.arange(3.0))
    assert jnp.array_equal(result.states[0, :, 0], jnp.array([2.0, 3.0, 4.0]))
    assert jnp.array_equal(result.observations[0, :, 0], result.states[0, :, 0])
    assert observation(jnp.array([1.0, 2.0]), None, 0).event_shape == (2,)


def test_event_dim_override_supports_batched_scalar_values():
    initial = dsx.DiracInitialCondition(jnp.array([1.0, 2.0]), event_dim=0)
    evolution = dsx.DiracStateEvolution(lambda x, u, *_: x + 1, event_dim=0)
    observation = dsx.DiracObservation(lambda x, u, t: x, event_dim=0)

    assert isinstance(initial, dist.Delta)
    assert initial.batch_shape == (2,)
    assert initial.event_shape == ()
    assert evolution(jnp.array([1.0, 2.0]), None, 0, 1).event_shape == ()
    assert observation(jnp.array([1.0, 2.0]), None, 0).event_shape == ()
