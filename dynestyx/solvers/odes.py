"""ODE solver backend for simulators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import diffrax as dfx
import jax.numpy as jnp
from jax import Array, lax

from dynestyx.types import State


def solve_ode(
    dynamics: Any,
    t0: float | Array,
    saveat_times: Array,
    x0: State,
    control_path_eval: Callable[[Array], Array | None],
    diffeqsolve_settings: dict[str, Any],
) -> Array:
    """Solve one ODE trajectory with Diffrax and save at requested times."""
    t1 = saveat_times[-1]

    def _early_return() -> Array:
        return jnp.broadcast_to(x0, (len(saveat_times),) + jnp.shape(x0))

    def _solve() -> Array:
        def _drift(t, y, args):
            u_t = args(t) if args is not None else None
            return dynamics.state_evolution.total_drift(x=y, u=u_t, t=t)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(_drift),
            t0=t0,
            t1=t1,
            y0=x0,
            saveat=dfx.SaveAt(ts=saveat_times),
            args=control_path_eval,
            **diffeqsolve_settings,
        )
        return sol.ys

    return lax.cond(t0 >= t1, _early_return, _solve)
