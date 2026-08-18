"""ODE solver backend for simulators."""

from __future__ import annotations

from typing import Any, cast

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Real

from dynestyx.models import DeterministicContinuousTimeStateEvolution, DynamicalModel
from dynestyx.types import as_scalar_time_array
from dynestyx.utils import _build_control_path_eval


def default_ode_diffeqsolve_settings() -> dict[str, Any]:
    """Return default settings for deterministic ODE path solves."""
    return {
        "solver": dfx.Tsit5(),
        "stepsize_controller": dfx.ConstantStepSize(),
        "adjoint": dfx.RecursiveCheckpointAdjoint(),
        "dt0": jnp.asarray(1e-3),
        "max_steps": 100_000,
    }


def solve_ode_interval(
    state_evolution: DeterministicContinuousTimeStateEvolution,
    *,
    initial_state: Real[Array, " state_dim"] | Real[Array, ""],
    t0: float | int | Real[Array, ""],
    t1: float | int | Real[Array, ""],
    u: Real[Array, " control_dim"] | Real[Array, ""] | None,
    diffeqsolve_settings: dict[str, Any],
) -> Real[Array, " state_dim"] | Real[Array, ""]:
    """Integrate one ODE interval with a control held constant."""
    state_dtype = jnp.result_type(jnp.asarray(initial_state), 0.0)
    t0_arr = as_scalar_time_array(t0, name="t0", dtype=state_dtype)
    t1_arr = as_scalar_time_array(t1, name="t1", dtype=state_dtype)
    t1_arr = eqx.error_if(
        t1_arr,
        t1_arr <= t0_arr,
        "ODE flow intervals require t1 > t0.",
    )

    def _drift(t, y, args):
        return state_evolution.total_drift(x=y, u=args, t=t)

    solution = dfx.diffeqsolve(
        dfx.ODETerm(_drift),
        t0=t0_arr,
        t1=t1_arr,
        y0=initial_state,
        saveat=dfx.SaveAt(t1=True),
        args=u,
        **diffeqsolve_settings,
    )
    return solution.ys[0]


def solve_ode_state_path(
    dynamics: DynamicalModel,
    *,
    initial_state: Real[Array, " state_dim"] | Real[Array, ""],
    t0: float | int | Real[Array, ""],
    path_times: Real[Array, " path_time"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None = None,
    diffeqsolve_settings: dict[str, Any] | None = None,
) -> Real[Array, "path_time state_dim"] | Real[Array, " path_time"]:
    """Solve one ODE state path with shared controls and default settings."""
    control_path_eval = _build_control_path_eval(ctrl_times, ctrl_values, path_times)
    settings = (
        diffeqsolve_settings
        if diffeqsolve_settings is not None
        else default_ode_diffeqsolve_settings()
    )
    t0_arr = as_scalar_time_array(t0, name="t0", dtype=path_times.dtype)
    t1 = path_times[-1]
    state_evolution = cast(
        DeterministicContinuousTimeStateEvolution,
        dynamics.state_evolution,
    )

    def _early_return():
        return jnp.broadcast_to(
            initial_state, (len(path_times),) + jnp.shape(initial_state)
        )

    def _solve():
        def _drift(t, y, args):
            u_t = args(t) if args is not None else None
            return state_evolution.total_drift(x=y, u=u_t, t=t)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(_drift),
            t0=t0_arr,
            t1=t1,
            y0=initial_state,
            saveat=dfx.SaveAt(ts=path_times),
            args=control_path_eval,
            **settings,
        )
        return sol.ys

    return lax.cond(t0_arr >= t1, _early_return, _solve)
