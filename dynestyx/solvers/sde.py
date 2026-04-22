"""SDE solver backends for SDESimulator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array, lax

from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.types import State, Time, TimeLike, as_scalar_time_array


def _apply_diffusion(diffusion_term: Array, dw: Array) -> Array:
    """Apply diffusion tensor/vector/scalar to Brownian increment."""
    if diffusion_term.ndim == 0:
        return diffusion_term * dw[0]
    if diffusion_term.ndim == 1:
        if dw.shape[0] == 1:
            return diffusion_term * dw[0]
        return diffusion_term * dw
    return diffusion_term @ dw


def _early_return_states(x0: State, saveat_times: Array) -> Array:
    """Repeat initial state when no forward integration is required."""
    return jnp.broadcast_to(x0, (len(saveat_times),) + jnp.shape(x0))


def _solve_sde_scan(
    dynamics: DynamicalModel,
    t0: Time,
    saveat_times: Array,
    x0: State,
    control_path_eval: Callable[[Array], Array | None],
    dt0: Time,
    *,
    key: Array | None,
) -> Array:
    if key is None:
        raise ValueError("PRNG key is required for em_scan SDE solves.")
    if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
        raise TypeError(
            "SDE solver requires ContinuousTimeStateEvolution, got "
            f"{type(dynamics.state_evolution)}"
        )

    state_evolution = dynamics.state_evolution
    bm_dim = int(state_evolution.bm_dim) if state_evolution.bm_dim is not None else 1

    dt0 = eqx.error_if(dt0, dt0 <= 0, f"EM scan requires dt0 > 0, got dt0={dt0!r}.")

    def _integrate_to_target(
        x_init: Array, t_init: Array, key_init: Array, t_target: Array
    ):
        def _cond_fn(carry):
            _, t_curr, _, t_end = carry
            return t_curr < t_end

        def _body_fn(carry):
            x_curr, t_curr, key_curr, t_end = carry
            h = jnp.minimum(dt0, t_end - t_curr)
            key_curr, k_step = jr.split(key_curr)
            z = jr.normal(k_step, shape=(bm_dim,), dtype=jnp.asarray(x_curr).dtype)
            dw = jnp.sqrt(h) * z
            u_t = control_path_eval(t_curr) if control_path_eval is not None else None
            drift = state_evolution.total_drift(x=x_curr, u=u_t, t=t_curr)
            diffusion = jnp.asarray(
                state_evolution.diffusion_coefficient(x=x_curr, u=u_t, t=t_curr)
            )
            x_next = x_curr + drift * h + _apply_diffusion(diffusion, dw)
            t_next = t_curr + h
            return x_next, t_next, key_curr, t_end

        carry0 = (x_init, t_init, key_init, t_target)
        x_out, t_out, key_out, _ = lax.while_loop(_cond_fn, _body_fn, carry0)
        return x_out, t_out, key_out

    def _scan_step(carry, t_target):
        x_prev, t_prev, key_prev = carry

        def _do_integrate(args):
            x_i, t_i, k_i = args
            return _integrate_to_target(x_i, t_i, k_i, t_target)

        def _skip_integrate(args):
            return args

        x_next, t_next, key_next = lax.cond(
            t_target > t_prev,
            _do_integrate,
            _skip_integrate,
            (x_prev, t_prev, key_prev),
        )
        return (x_next, t_next, key_next), x_next

    (_, _, _), states = lax.scan(_scan_step, (x0, t0, key), saveat_times)
    return states


def _solve_sde_diffrax(
    dynamics: DynamicalModel,
    t0: Time,
    saveat_times: Array,
    x0: State,
    control_path_eval: Callable[[Array], Array | None],
    diffeqsolve_settings: dict[str, Any],
    *,
    key: Array | None,
    tol_vbt: Time,
) -> Array:
    if key is None:
        raise ValueError("PRNG key is required for diffrax SDE solves.")
    if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
        raise TypeError(
            "SDE solver requires ContinuousTimeStateEvolution, got "
            f"{type(dynamics.state_evolution)}"
        )

    def _drift(t, y, args):
        u_t = args(t) if args is not None else None
        return dynamics.state_evolution.total_drift(x=y, u=u_t, t=t)

    def _diffusion(t, y, args):
        u_t = args(t) if args is not None else None
        return dynamics.state_evolution.diffusion_coefficient(x=y, u=u_t, t=t)

    k_bm, _ = jr.split(key, 2)
    bm = dfx.VirtualBrownianTree(
        t0=t0,
        t1=saveat_times[-1],
        tol=tol_vbt,
        shape=(dynamics.state_evolution.bm_dim,),
        key=k_bm,
    )
    terms = dfx.MultiTerm(  # type: ignore[arg-type]
        dfx.ODETerm(_drift), dfx.ControlTerm(_diffusion, bm)
    )
    sol = dfx.diffeqsolve(
        terms,
        t0=t0,
        t1=saveat_times[-1],
        y0=x0,
        saveat=dfx.SaveAt(ts=saveat_times),
        args=control_path_eval,
        **diffeqsolve_settings,
    )
    return sol.ys


def solve_sde(
    *,
    source: Literal["diffrax", "em_scan"],
    dynamics: DynamicalModel,
    t0: TimeLike,
    saveat_times: Array,
    x0: State,
    control_path_eval: Callable[[Array], Array | None],
    diffeqsolve_settings: dict[str, Any],
    key: Array,
    tol_vbt: TimeLike | None = None,
) -> Array:
    """Dispatch between SDE solver backends used by SDESimulator."""

    t0_arr = as_scalar_time_array(t0, name="t0", dtype=saveat_times.dtype)
    needs_integration = t0_arr < saveat_times[-1]

    def _do_solve(_: Array) -> Array:
        if source == "diffrax":
            if tol_vbt is None:
                raise ValueError("tol_vbt is required when source='diffrax'.")
            tol_vbt_arr = as_scalar_time_array(
                tol_vbt, name="tol_vbt", dtype=saveat_times.dtype
            )
            return _solve_sde_diffrax(
                dynamics,
                t0_arr,
                saveat_times,
                x0,
                control_path_eval,
                diffeqsolve_settings,
                key=key,
                tol_vbt=tol_vbt_arr,
            )

        if source == "em_scan":
            dt0_like = diffeqsolve_settings.get("dt0")
            if dt0_like is None:
                raise ValueError("dt0 is required when source='em_scan'.")
            dt0_arr = as_scalar_time_array(
                dt0_like, name="dt0", dtype=saveat_times.dtype
            )
            return _solve_sde_scan(
                dynamics,
                t0_arr,
                saveat_times,
                x0,
                control_path_eval,
                dt0_arr,
                key=key,
            )

        raise ValueError(f"Unknown SDE solver source: {source}")

    def _do_early_return(_: Array) -> Array:
        return _early_return_states(x0, saveat_times)

    return lax.cond(needs_integration, _do_solve, _do_early_return, t0_arr)
