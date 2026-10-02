"""Minimal coverage for shared Diffrax SDE interval integration."""

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import pytest

import dynestyx as dsx
from dynestyx.solvers import solve_diffrax_sde_interval, solve_sde_state_path


@pytest.mark.parametrize("solver", [dfx.Euler(), dfx.Heun()])
def test_brownian_increment_solver_runs_for_path_and_interval(solver):
    dynamics = dsx.LTI_continuous(
        A=jnp.array([[-0.5]]),
        L=jnp.array([[0.2]]),
        H=jnp.array([[1.0]]),
        R=jnp.array([[0.1]]),
    )
    state_evolution = dynamics.state_evolution
    assert isinstance(
        state_evolution,
        dsx.StochasticContinuousTimeStateEvolution,
    )
    settings = {
        "solver": solver,
        "stepsize_controller": dfx.ConstantStepSize(),
        "adjoint": dfx.RecursiveCheckpointAdjoint(),
        "dt0": jnp.array(0.01),
        "max_steps": 1_000,
    }
    key = jr.PRNGKey(4)

    endpoint = solve_diffrax_sde_interval(
        state_evolution,
        initial_state=jnp.array([0.0]),
        t0=0.0,
        t1=0.1,
        u=None,
        diffeqsolve_settings=settings,
        key=key,
        tol_vbt=0.005,
    )
    path = solve_sde_state_path(
        dynamics,
        source="diffrax",
        initial_state=jnp.array([0.0]),
        t0=0.0,
        path_times=jnp.array([0.05, 0.1]),
        ctrl_times=None,
        ctrl_values=None,
        diffeqsolve_settings=settings,
        key=key,
        tol_vbt=0.005,
    )

    assert endpoint.shape == (1,)
    assert path.shape == (2, 1)
    assert jnp.all(jnp.isfinite(endpoint))
    assert jnp.all(jnp.isfinite(path))
