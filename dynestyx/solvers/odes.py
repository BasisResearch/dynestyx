"""ODE solver backend for simulators."""

from __future__ import annotations

from typing import Any

import diffrax as dfx
import jax.numpy as jnp
from jax import Array, lax

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


def solve_ode_state_path(
    dynamics: Any,
    *,
    initial_state: Array,
    t0: float | int | Array,
    path_times: Array,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    diffeqsolve_settings: dict[str, Any] | None = None,
) -> Array:
    """Solve one ODE state path with shared controls and default settings."""
    control_path_eval = _build_control_path_eval(ctrl_times, ctrl_values, path_times)
    settings = (
        diffeqsolve_settings
        if diffeqsolve_settings is not None
        else default_ode_diffeqsolve_settings()
    )
    t0_arr = as_scalar_time_array(t0, name="t0", dtype=path_times.dtype)
    t1 = path_times[-1]

    def _early_return() -> Array:
        return jnp.broadcast_to(
            initial_state, (len(path_times),) + jnp.shape(initial_state)
        )

    def _solve() -> Array:
        def _drift(t, y, args):
            u_t = args(t) if args is not None else None
            return dynamics.state_evolution.total_drift(x=y, u=u_t, t=t)

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
