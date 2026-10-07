"""Result shape normalization preserves initial distributions and dynamics inputs."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed

import dynestyx as dsx


@pytest.mark.parametrize("vector", [False, True])
@pytest.mark.parametrize("length", [1, 3])
def test_ode_preserves_initial_state_shape_but_normalizes_results(vector, length):
    initial = (
        dist.Normal(jnp.zeros(1), 1.0).to_event(1) if vector else dist.Normal(0.0, 1.0)
    )
    model = dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=dsx.ContinuousTimeStateEvolution(drift=lambda x, u, t: -x),
        observation_model=lambda x, u, t: dist.Normal(jnp.sum(x), 0.2),
    )
    assert model.initial_condition is initial

    def drift(x, u, t):
        assert x.shape == ((1,) if vector else ())
        return -x

    model = eqx.tree_at(lambda m: m.state_evolution.drift, model, drift)
    result = jax.jit(
        lambda key: dsx.simulate(
            model,
            rng_key=key,
            predict_times=jnp.arange(float(length)) * 0.1,
            n_simulations=2,
        )
    )(jr.key(0))
    assert result.x_0.shape == (2, 1)
    assert result.states.shape == (2, length, 1)
    assert result.observations.shape == (2, length, 1)
    assert jnp.allclose(result.states[:, 0], result.x_0)


@pytest.mark.parametrize("members", [1, 3])
def test_plated_scalar_ode_results_have_coordinate_axis(members):
    initial = dist.Normal(jnp.arange(float(members)), 0.1)
    with dsx.ODESimulator(n_simulations=2), seed(rng_seed=0):
        with dsx.plate("members", members):
            model = dsx.DynamicalModel(
                initial_condition=initial,
                state_evolution=dsx.ContinuousTimeStateEvolution(
                    drift=lambda x, u, t: -x
                ),
                observation_model=lambda x, u, t: dist.Normal(x, 0.2),
            )
            assert model.initial_condition is initial
            assert model.initial_condition.event_shape == ()
            result = dsx.sample("f", model, predict_times=jnp.array([0.0, 0.1]))
    assert result.x_0.shape == (members, 2, 1)
    assert result.states.shape == (members, 2, 2, 1)
    assert result.observations.shape == (members, 2, 2, 1)


def test_sde_preserves_vector_ic_and_normalizes_scalar_observations():
    initial = dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1))
    model = dsx.DynamicalModel(
        initial_condition=initial,
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -x,
            diffusion=dsx.ScalarDiffusion(0.1, bm_dim=1),
        ),
        observation_model=lambda x, u, t: dist.Normal(x[0], 0.2),
    )
    assert model.initial_condition is initial
    result = dsx.simulate(
        model,
        rng_key=jr.key(0),
        predict_times=jnp.array([0.0, 0.1]),
        n_simulations=2,
    )
    assert result.x_0 is not None and result.x_0.shape == (2, 1)
    assert result.states is not None and result.states.shape == (2, 2, 1)
    assert result.observations is not None and result.observations.shape == (2, 2, 1)


@pytest.mark.parametrize("categorical", [False, True])
@pytest.mark.parametrize("length", [1, 3])
def test_discrete_scalar_internals_are_unchanged_but_results_have_event_axis(
    categorical, length
):
    initial = (
        dist.Categorical(probs=jnp.array([0.4, 0.6]))
        if categorical
        else dist.Normal(0.0, 1.0)
    )

    def transition(x, u, t_now, t_next):
        assert x.ndim == 0
        if categorical:
            return dist.Categorical(probs=jnp.array([[0.8, 0.2], [0.3, 0.7]])[x])
        return dist.Normal(0.0, 1.0)

    model = dsx.DynamicalModel(
        initial_condition=initial,
        # Noncategorical construction probes use a vector even for scalar ICs.
        state_evolution=lambda x, u, t_now, t_next: transition(
            jnp.squeeze(x), u, t_now, t_next
        ),
        observation_model=lambda x, u, t: dist.Normal(jnp.asarray(x, dtype=float), 0.2),
    )
    assert model.initial_condition is initial
    result = dsx.simulate(
        model,
        rng_key=jr.key(0),
        predict_times=jnp.arange(float(length)),
        n_simulations=2,
    )
    assert result.x_0 is not None and result.x_0.shape == (2, 1)
    assert result.states is not None and result.states.shape == (2, length, 1)
    assert result.observations is not None and result.observations.shape == (
        2,
        length,
        1,
    )
