"""Tests for shared continuous-time state-path solver wrappers."""

import warnings

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


def _make_imex_dynamics(*, use_imex_drift: bool) -> dsx.DynamicalModel:
    drift = (
        dsx.ImExDrift(
            explicit_term=lambda x, u, t: -0.7 * x,
            implicit_term=lambda x, u, t: -1.3 * x,
        )
        if use_imex_drift
        else (lambda x, u, t: -0.7 * x)
    )
    return dsx.DynamicalModel(
        control_dim=0,
        initial_condition=dist.Delta(jnp.array([1.0]), event_dim=1),
        state_evolution=dsx.ContinuousTimeStateEvolution(drift=drift),
        observation_model=lambda x, u, t: dist.Delta(x, event_dim=1),
    )


def test_solve_ode_state_path_imex_solver_matches_analytic_solution():
    # dx/dt = -0.7*x (explicit) + -1.3*x (implicit) => x(t) = x0 * exp(-2*t)
    path_times = jnp.array([0.0, 0.5, 1.0])
    states = solve_ode_state_path(
        _make_imex_dynamics(use_imex_drift=True),
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


def test_solve_ode_state_path_imex_solver_does_not_warn():
    path_times = jnp.array([0.0, 0.5, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solve_ode_state_path(
            _make_imex_dynamics(use_imex_drift=True),
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


def test_solve_ode_state_path_imex_solver_requires_imex_drift():
    path_times = jnp.array([0.0, 1.0])
    with pytest.raises(ValueError, match="ImExDrift"):
        solve_ode_state_path(
            _make_imex_dynamics(use_imex_drift=False),
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
def test_solve_ode_state_path_imex_drift_with_non_imex_solver_warns(t0):
    # The mismatch check runs eagerly in solve_ode_state_path, before
    # lax.cond dispatch, so the warning fires regardless of t0 (i.e. even in
    # the early-return branch, which never calls diffeqsolve).
    path_times = jnp.array([0.0, 1.0])
    with pytest.warns(UserWarning, match="ImExDrift") as record:
        solve_ode_state_path(
            _make_imex_dynamics(use_imex_drift=True),
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

    assert "Tsit5" in str(record[0].message)


def test_dynamical_model_rejects_potential_with_imex_drift():
    with pytest.raises(ValueError, match="potential"):
        dsx.DynamicalModel(
            control_dim=0,
            initial_condition=dist.Delta(jnp.array([1.0]), event_dim=1),
            state_evolution=dsx.ContinuousTimeStateEvolution(
                drift=dsx.ImExDrift(
                    explicit_term=lambda x, u, t: -0.7 * x,
                    implicit_term=lambda x, u, t: -1.3 * x,
                ),
                potential=lambda x, u, t: jnp.sum(x**2),
            ),
            observation_model=lambda x, u, t: dist.Delta(x, event_dim=1),
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
