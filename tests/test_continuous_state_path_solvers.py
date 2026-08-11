"""Tests for shared continuous-time state-path solver wrappers."""

import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
import pytest

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


def _make_imex_dynamics(*, implicit_drift=None) -> dsx.DynamicalModel:
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Delta(jnp.array([1.0]), event_dim=1),
        state_evolution=dsx.ContinuousTimeStateEvolution(
            drift=lambda x, u, t: -0.7 * x,
            implicit_drift=implicit_drift,
        ),
        observation_model=lambda x, u, t: dist.Delta(x, event_dim=1),
    )


def test_solve_ode_state_path_imex_solver_matches_analytic_solution():
    # dx/dt = -0.7*x (explicit) + -1.3*x (implicit) => x(t) = x0 * exp(-2*t)
    path_times = jnp.array([0.0, 0.5, 1.0])
    states = solve_ode_state_path(
        _make_imex_dynamics(implicit_drift=lambda x, u, t: -1.3 * x),
        initial_state=jnp.array([1.0]),
        t0=jnp.array(0.0),
        path_times=path_times,
        diffeqsolve_settings={
            "solver": dfx.KenCarp4(),
            "stepsize_controller": dfx.PIDController(rtol=1e-6, atol=1e-6),
            "adjoint": dfx.RecursiveCheckpointAdjoint(),
            "dt0": jnp.array(0.01),
            "max_steps": 10_000,
        },
    )

    expected = jnp.exp(-2.0 * path_times)
    assert jnp.allclose(states[:, 0], expected, atol=1e-3)


def test_solve_ode_state_path_imex_solver_requires_implicit_drift():
    path_times = jnp.array([0.0, 1.0])
    with pytest.raises(ValueError, match="implicit_drift"):
        solve_ode_state_path(
            _make_imex_dynamics(implicit_drift=None),
            initial_state=jnp.array([1.0]),
            t0=jnp.array(0.0),
            path_times=path_times,
            diffeqsolve_settings={
                "solver": dfx.KenCarp4(),
                "stepsize_controller": dfx.PIDController(rtol=1e-6, atol=1e-6),
                "adjoint": dfx.RecursiveCheckpointAdjoint(),
                "dt0": jnp.array(0.01),
                "max_steps": 10_000,
            },
        )


@pytest.mark.parametrize("t0", [0.0, 1.0])
def test_solve_ode_state_path_implicit_drift_requires_imex_solver(t0):
    # Regression check: the mismatch must raise even when t0 >= t1, i.e. inside
    # the lax.cond branch that would otherwise skip straight to early-return
    # without ever calling diffeqsolve.
    path_times = jnp.array([0.0, 1.0])
    with pytest.raises(ValueError, match="IMEX solver"):
        solve_ode_state_path(
            _make_imex_dynamics(implicit_drift=lambda x, u, t: -1.3 * x),
            initial_state=jnp.array([1.0]),
            t0=jnp.array(t0),
            path_times=path_times,
            diffeqsolve_settings={
                "solver": dfx.Tsit5(),
                "stepsize_controller": dfx.ConstantStepSize(),
                "adjoint": dfx.RecursiveCheckpointAdjoint(),
                "dt0": jnp.array(0.01),
                "max_steps": 1_000,
            },
        )


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
