"""Simulation results retain a trailing coordinate axis."""

import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed

import dynestyx as dsx
from dynestyx.control import DiscreteControlLoopSimulator


@pytest.mark.parametrize("kind", ["ode", "sde", "discrete", "categorical"])
def test_simulation_result_shapes(kind):
    initial = dist.Normal(0.0, 1.0)
    if kind == "sde":
        initial = dist.Normal(jnp.zeros(1), 1.0).to_event(1)
    elif kind == "categorical":
        initial = dist.Categorical(probs=jnp.array([0.4, 0.6]))

    if kind in ("ode", "sde"):
        evolution = dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -x,
            diffusion=dsx.ScalarDiffusion(0.1, bm_dim=1) if kind == "sde" else None,
        )
    else:
        evolution = lambda x, u, t_now, t_next: initial

    model = dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=evolution,
        observation_model=lambda x, u, t: dist.Normal(jnp.sum(x), 0.2),
    )
    result = dsx.simulate(
        model,
        rng_key=jr.key(0),
        predict_times=jnp.array([0.0, 0.1, 0.2]),
        n_simulations=2,
    )
    assert result.x_0 is not None and result.x_0.shape == (2, 1)
    assert result.states is not None and result.states.shape == (2, 3, 1)
    assert result.observations is not None and result.observations.shape == (2, 3, 1)


@pytest.mark.parametrize(
    "state_dim", [None, 1, 2], ids=["scalar", "vector1", "vector2"]
)
def test_control_loop_simulation_result_shapes(state_dim):
    initial = (
        dist.Normal(0.0, 1.0)
        if state_dim is None
        else dist.Normal(jnp.zeros(state_dim), 1.0).to_event(1)
    )
    model = dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=lambda x, u, t_now, t_next: initial,
        observation_model=lambda x, u, t: dist.Normal(jnp.sum(x), 0.2),
        control_dim=1,
        observation_control_alignment="previous_transition",
    )
    with (
        DiscreteControlLoopSimulator(
            control_policy=lambda x_hat, t_now, t_next, s: (jnp.zeros(1), s),
            use_true_state=True,
        ),
        seed(rng_seed=0),
    ):
        result = dsx.sample("f", model, predict_times=jnp.array([0.0, 0.1, 0.2]))

    coordinate_dim = state_dim or 1
    assert result.x_0.shape == (1, coordinate_dim)
    assert result.states.shape == (1, 3, coordinate_dim)
    assert result.observations.shape == (1, 2, 1)
    assert result.controls.shape == (1, 2, 1)


def test_plated_simulation_result_shapes():
    with dsx.ODESimulator(n_simulations=2), seed(rng_seed=0), dsx.plate("members", 3):
        model = dsx.DynamicalModel(
            initial_condition=dist.Normal(jnp.arange(3.0), 0.1),
            state_dim=1,
            state_evolution=dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: -x),
            observation_model=lambda x, u, t: dist.Normal(x, 0.2),
        )
        result = dsx.sample("f", model, predict_times=jnp.array([0.0, 0.1]))
    assert result.x_0.shape == (3, 2, 1)
    assert result.states.shape == (3, 2, 2, 1)
    assert result.observations.shape == (3, 2, 2, 1)
