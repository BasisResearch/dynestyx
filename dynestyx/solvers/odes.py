"""ODE solver backend for simulators."""

from __future__ import annotations

from typing import Any, cast, get_origin

import diffrax as dfx
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


def _solver_needs_multi_term(solver: dfx.AbstractSolver) -> bool:
    """Whether `solver` requires diffrax's `MultiTerm` (e.g. IMEX solvers like
    `KenCarp3/4/5`, `Sil3`), rather than a single `ODETerm`.

    Mirrors the check diffrax itself performs (see
    `diffrax._solver.runge_kutta.AbstractRungeKutta.__init_subclass__` and
    `diffrax._integrate._assert_term_compatible`), so it covers any current or
    future diffrax solver without hardcoding solver names.
    """
    return get_origin(solver.term_structure) is dfx.MultiTerm


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

    needs_multi_term = _solver_needs_multi_term(settings["solver"])
    has_implicit_drift = state_evolution.implicit_drift is not None
    if has_implicit_drift and not needs_multi_term:
        raise ValueError(
            "You supplied an `implicit_drift`, but did not select an IMEX "
            "solver (one whose term_structure requires diffrax's MultiTerm, "
            "e.g. diffrax.KenCarp4()). Remove `implicit_drift` or choose an "
            "IMEX solver."
        )
    if needs_multi_term and not has_implicit_drift:
        raise ValueError(
            "Solver requires separate explicit/implicit terms (diffrax "
            "MultiTerm), but `implicit_drift` is not set on this "
            "ContinuousTimeStateEvolution. Set `implicit_drift` to the "
            "stiff component of the vector field."
        )

    def _early_return():
        return jnp.broadcast_to(
            initial_state, (len(path_times),) + jnp.shape(initial_state)
        )

    def _solve():
        if needs_multi_term:

            def _explicit(t, y, args):
                u_t = args(t) if args is not None else None
                return state_evolution.total_drift(x=y, u=u_t, t=t)

            def _implicit(t, y, args):
                u_t = args(t) if args is not None else None
                assert state_evolution.implicit_drift is not None
                return state_evolution.implicit_drift(x=y, u=u_t, t=t)

            terms: dfx.AbstractTerm = dfx.MultiTerm(
                dfx.ODETerm(_explicit), dfx.ODETerm(_implicit)
            )
        else:

            def _drift(t, y, args):
                u_t = args(t) if args is not None else None
                return state_evolution.total_drift(x=y, u=u_t, t=t)

            terms = dfx.ODETerm(_drift)

        sol = dfx.diffeqsolve(
            terms,
            t0=t0_arr,
            t1=t1,
            y0=initial_state,
            saveat=dfx.SaveAt(ts=path_times),
            args=control_path_eval,
            **settings,
        )
        return sol.ys

    return lax.cond(t0_arr >= t1, _early_return, _solve)
