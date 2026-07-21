"""Tests for shared continuous-time state-path solver wrappers."""

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx.solvers import (
    solve_ode_state_path,
    solve_sde_state_path,
)


def _make_controlled_dynamics(*, stochastic: bool) -> dsx.DynamicalModel:
    diffusion = (
        dsx.FullDiffusion(lambda x, u, t: jnp.zeros((1, 1))) if stochastic else None
    )
    return dsx.DynamicalModel(
        control_dim=1,
        initial_condition=dist.Delta(jnp.array([0.0]), event_dim=1),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: u,
            diffusion=diffusion,
        ),
        observation_model=lambda x, u, t: dist.Delta(x, event_dim=1),
    )


def test_solve_ode_state_path_applies_controls():
    path_times = jnp.array([0.0, 0.5, 1.0])
    states = solve_ode_state_path(
        _make_controlled_dynamics(stochastic=False),
        initial_state=jnp.array([0.0]),
        t0=jnp.array(0.0),
        path_times=path_times,
        ctrl_times=path_times,
        ctrl_values=jnp.full((3, 1), 2.0),
        diffeqsolve_settings={
            "solver": dfx.Tsit5(),
            "stepsize_controller": dfx.ConstantStepSize(),
            "adjoint": dfx.RecursiveCheckpointAdjoint(),
            "dt0": jnp.array(0.01),
            "max_steps": 1_000,
        },
    )

    assert jnp.allclose(states[:, 0], jnp.array([0.0, 1.0, 2.0]), atol=1e-5)


def test_solve_sde_state_path_applies_controls():
    path_times = jnp.array([0.0, 0.5, 1.0])
    states = solve_sde_state_path(
        _make_controlled_dynamics(stochastic=True),
        source="em_scan",
        initial_state=jnp.array([0.0]),
        t0=jnp.array(0.0),
        path_times=path_times,
        ctrl_times=path_times,
        ctrl_values=jnp.full((3, 1), 2.0),
        diffeqsolve_settings={"dt0": jnp.array(0.01)},
        key=jr.PRNGKey(0),
    )

    assert jnp.allclose(states[:, 0], jnp.array([0.0, 1.0, 2.0]), atol=1e-5)
