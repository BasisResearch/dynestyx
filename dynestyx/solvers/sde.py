"""SDE solver backends for SDESimulator."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import diffrax as dfx
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jax import Array, lax, vmap

from dynestyx.models import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.models.diffusions import (
    EvaluatedDiffusion,
    apply_diffusion,
    diffusion_as_matrix,
    diffusion_covariance,
    evaluate_diffusion,
)
from dynestyx.types import State, Time, TimeLike, as_scalar_time_array


def _early_return_states(x0: State, saveat_times: Array) -> Array:
    """Build no-op solve output by repeating the initial state.

    Args:
        x0: Initial state.
        saveat_times: Requested output times.

    Returns:
        Array of states with `x0` repeated across the time axis.
    """
    return jnp.broadcast_to(x0, (len(saveat_times),) + jnp.shape(x0))


def _require_bm_dim(state_evolution: ContinuousTimeStateEvolution) -> int:
    """Return Brownian dimension or raise if unspecified.

    Args:
        state_evolution: Continuous-time state evolution.

    Returns:
        Brownian motion dimension used by EM sampling.
    """
    if state_evolution.bm_dim is None:
        raise ValueError("SDE sampling requires state_evolution.bm_dim to be set.")
    return int(state_evolution.bm_dim)


def _require_diffusion_spec(
    state_evolution: ContinuousTimeStateEvolution,
) -> Any:
    """Get diffusion spec or raise if unavailable.

    Args:
        state_evolution: Continuous-time state evolution.

    Returns:
        Diffusion callable or constant spec.
    """
    diffusion_spec = state_evolution.diffusion_coefficient
    if diffusion_spec is None:
        raise ValueError("SDE solver requires diffusion_coefficient to be defined.")
    return diffusion_spec


def _em_local_terms(
    state_evolution: ContinuousTimeStateEvolution,
    x: Array,
    u: Array | None,
    t_now: Array,
) -> tuple[Array, EvaluatedDiffusion]:
    """Compute local EM drift and diffusion terms.

    Args:
        state_evolution: Continuous-time state evolution.
        x: Current state.
        u: Optional control at `t_now`.
        t_now: Current time.

    Returns:
        Tuple `(drift, diffusion)` at `(x, u, t_now)`.
    """
    drift = state_evolution.total_drift(x=x, u=u, t=t_now)
    diffusion_spec = _require_diffusion_spec(state_evolution)
    diffusion = evaluate_diffusion(
        diffusion_spec,
        diffusion_type=state_evolution.diffusion_type,
        bm_dim=state_evolution.bm_dim,
        x=x,
        u=u,
        t=t_now,
        state_dim=x.shape[-1],
    )
    return drift, diffusion


def _em_moments_from_terms(
    x: Array, dt: Array, drift: Array, diffusion: EvaluatedDiffusion
) -> tuple[Array, Array]:
    """Convert local EM terms to one-step Gaussian moments.

    Args:
        x: Current state.
        dt: Step size.
        drift: Drift evaluated at the current point.
        diffusion: Diffusion coefficient evaluated at the current point.

    Returns:
        Tuple `(loc, cov)` for the EM Gaussian approximation.
    """
    loc = x + drift * dt
    cov = diffusion_covariance(diffusion, state_dim=x.shape[-1]) * dt
    return loc, cov


def _em_sample_from_terms(
    x: Array,
    dt: Array,
    drift: Array,
    diffusion: EvaluatedDiffusion,
    *,
    key: Array,
    bm_dim: int,
) -> tuple[Array, Array]:
    """Sample one EM next-state from local drift/diffusion terms.

    Args:
        x: Current state.
        dt: Step size.
        drift: Drift evaluated at the current point.
        diffusion: Diffusion coefficient evaluated at the current point.
        key: PRNG key for the Brownian increment.
        bm_dim: Brownian motion dimension.

    Returns:
        Tuple `(x_next, key_next)` after one sampled EM step.
    """
    key_next, k_step = jr.split(key)
    z = jr.normal(k_step, shape=(bm_dim,), dtype=jnp.asarray(x).dtype)
    dw = jnp.sqrt(dt) * z
    x_next = x + drift * dt + apply_diffusion(diffusion, dw, state_dim=x.shape[-1])
    return x_next, key_next


def euler_maruyama_integrate_state_to_time(
    state_evolution: ContinuousTimeStateEvolution,
    x_init: Array,
    t_init: Time,
    key_init: Array,
    t_target: Time,
    *,
    dt0: Time,
    control_path_eval: Callable[[Array], Array | None] | None = None,
) -> tuple[Array, Array, Array]:
    """Integrate a sampled EM path from `t_init` to `t_target`.

    Args:
        state_evolution: Continuous-time state evolution.
        x_init: Initial state.
        t_init: Initial time as a scalar JAX array.
        key_init: Initial PRNG key.
        t_target: Target end time as a scalar JAX array.
        dt0: Maximum EM substep size as a scalar JAX array.
        control_path_eval: Optional control evaluator `u(t)`.

    Returns:
        Tuple `(x_out, t_out, key_out)` at integration end.
    """
    dt0 = eqx.error_if(
        dt0, dt0 <= 0, f"EM integration requires dt0 > 0, got dt0={dt0!r}."
    )

    bm_dim = _require_bm_dim(state_evolution)

    def _cond_fn(carry):
        _, t_curr, _, t_end = carry
        return t_curr < t_end

    def _body_fn(carry):
        x_curr, t_curr, key_curr, t_end = carry
        h = jnp.minimum(dt0, t_end - t_curr)
        u_t = control_path_eval(t_curr) if control_path_eval is not None else None
        drift, diffusion = _em_local_terms(state_evolution, x_curr, u_t, t_curr)
        x_next, key_next = _em_sample_from_terms(
            x_curr, h, drift, diffusion, key=key_curr, bm_dim=bm_dim
        )
        return x_next, t_curr + h, key_next, t_end

    carry0 = (x_init, t_init, key_init, t_target)
    x_out, t_out, key_out, _ = lax.while_loop(_cond_fn, _body_fn, carry0)
    return x_out, t_out, key_out


def euler_maruyama_loc_cov(
    state_evolution: ContinuousTimeStateEvolution,
    x: Array,
    u: Array | None,
    t_now: Array,
    t_next: Array,
) -> dict[str, Array]:
    """Compute one-step Euler-Maruyama transition moments.

    Args:
        state_evolution: Continuous-time state evolution with drift and diffusion.
        x: Current state(s). Supports either:
            - `(state_dim,)` for a single transition, or
            - `(..., state_dim)` for batched transitions, preserving leading dims.
        u: Optional control input(s). If provided, supports either:
            - `(control_dim,)` for a single transition, or
            - `(..., control_dim)` for batched transitions with leading dims
              matching `x`.
            Pass `None` for uncontrolled dynamics.
        t_now: Start time(s), scalar or shape matching the leading dims of `x`.
        t_next: End time(s), scalar or shape matching the leading dims of `x`.

    Returns:
        Dictionary with:
        - `"loc"`: predicted mean(s), shape `(state_dim,)` for single-transition
          input or `(..., state_dim)` for batched input.
        - `"cov"`: predicted covariance matrix/matrices, shape
          `(state_dim, state_dim)` for single-transition input or
          `(..., state_dim, state_dim)` for batched input.

    Batching behavior:
        Leading batch dimensions are preserved. For example, time-batched input
        of shape `(T, state_dim)` returns `(T, state_dim)` rather than moving
        the time axis to the front as a side effect of `vmap`.
    """
    x_arr = jnp.asarray(x)

    def _step_interval(_x, _u, _t_now, _t_next):
        drift, diffusion = _em_local_terms(state_evolution, _x, _u, _t_now)
        return _em_moments_from_terms(_x, _t_next - _t_now, drift, diffusion)

    if x_arr.ndim == 1:
        loc, cov = _step_interval(x_arr, u, jnp.asarray(t_now), jnp.asarray(t_next))
        return {"loc": loc, "cov": cov}

    batch_shape = x_arr.shape[:-1]
    state_dim = x_arr.shape[-1]

    if u is None:
        u_arr = None
    else:
        u_arr = jnp.asarray(u)
        if u_arr.ndim == 1 or u_arr.shape[:-1] != batch_shape:
            raise ValueError(
                "For batched x, u must be None or have leading dimensions "
                "matching x in euler_maruyama_loc_cov."
            )

    t_now_arr = jnp.asarray(t_now)
    if t_now_arr.ndim == 0:
        t_now_arr = jnp.broadcast_to(t_now_arr, batch_shape)
    elif t_now_arr.shape != batch_shape:
        raise ValueError(
            "t_now must be scalar or have shape matching x leading dimensions "
            "in euler_maruyama_loc_cov."
        )

    t_next_arr = jnp.asarray(t_next)
    if t_next_arr.ndim == 0:
        t_next_arr = jnp.broadcast_to(t_next_arr, batch_shape)
    elif t_next_arr.shape != batch_shape:
        raise ValueError(
            "t_next must be scalar or have shape matching x leading dimensions "
            "in euler_maruyama_loc_cov."
        )

    def _batched_step(_x, _u, _t_now, _t_next):
        if _x.ndim == 1:
            return _step_interval(_x, _u, _t_now, _t_next)
        in_axes_u = None if _u is None else 0
        return vmap(_batched_step, in_axes=(0, in_axes_u, 0, 0))(
            _x, _u, _t_now, _t_next
        )

    loc, cov = _batched_step(x_arr, u_arr, t_now_arr, t_next_arr)

    if loc.shape != x_arr.shape:
        raise ValueError(
            "euler_maruyama_loc_cov produced a location shape inconsistent with "
            f"the input state shape: x.shape={x_arr.shape}, loc.shape={loc.shape}."
        )
    expected_cov_shape = batch_shape + (state_dim, state_dim)
    if cov.shape != expected_cov_shape:
        raise ValueError(
            "euler_maruyama_loc_cov produced a covariance shape inconsistent with "
            f"the input state shape: expected {expected_cov_shape}, got {cov.shape}."
        )
    return {"loc": loc, "cov": cov}


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
    """Solve an SDE with fixed-step Euler-Maruyama scan integration.

    Args:
        dynamics: Dynamical model with continuous-time state evolution.
        t0: Initial time as a scalar JAX array.
        saveat_times: Times at which to return states.
        x0: Initial state.
        control_path_eval: Optional control evaluator `u(t)`.
        dt0: Maximum EM substep size as a scalar JAX array.
        key: PRNG key for Brownian increments.

    Returns:
        Simulated state trajectory at `saveat_times`.
    """
    if key is None:
        raise ValueError("PRNG key is required for em_scan SDE solves.")
    if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
        raise TypeError(
            "SDE solver requires ContinuousTimeStateEvolution, got "
            f"{type(dynamics.state_evolution)}"
        )

    state_evolution = dynamics.state_evolution
    dt0 = eqx.error_if(dt0, dt0 <= 0, f"EM scan requires dt0 > 0, got dt0={dt0!r}.")

    def _scan_step(carry, t_target):
        x_prev, t_prev, key_prev = carry

        def _do_integrate(args):
            x_i, t_i, k_i = args
            return euler_maruyama_integrate_state_to_time(
                state_evolution,
                x_i,
                t_i,
                k_i,
                t_target,
                dt0=dt0,
                control_path_eval=control_path_eval,
            )

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
    """Solve an SDE with Diffrax and a VirtualBrownianTree control.

    Args:
        dynamics: Dynamical model with continuous-time state evolution.
        t0: Initial time as a scalar JAX array.
        saveat_times: Times at which to return states.
        x0: Initial state.
        control_path_eval: Optional control evaluator `u(t)`.
        diffeqsolve_settings: Extra keyword arguments for `dfx.diffeqsolve`.
        key: PRNG key for Brownian tree construction.
        tol_vbt: VirtualBrownianTree tolerance as a scalar JAX array.

    Returns:
        Simulated state trajectory at `saveat_times`.
    """
    if key is None:
        raise ValueError("PRNG key is required for diffrax SDE solves.")
    if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
        raise TypeError(
            "SDE solver requires ContinuousTimeStateEvolution, got "
            f"{type(dynamics.state_evolution)}"
        )

    state_evolution = dynamics.state_evolution

    def _drift(t, y, args):
        u_t = args(t) if args is not None else None
        return state_evolution.total_drift(x=y, u=u_t, t=t)

    def _diffusion(t, y, args):
        u_t = args(t) if args is not None else None
        diffusion_spec = _require_diffusion_spec(state_evolution)
        diffusion = evaluate_diffusion(
            diffusion_spec,
            diffusion_type=state_evolution.diffusion_type,
            bm_dim=state_evolution.bm_dim,
            x=y,
            u=u_t,
            t=t,
            state_dim=y.shape[-1],
        )
        return diffusion_as_matrix(diffusion, state_dim=y.shape[-1])

    k_bm, _ = jr.split(key, 2)
    bm = dfx.VirtualBrownianTree(
        t0=t0,
        t1=saveat_times[-1],
        tol=tol_vbt,
        shape=(state_evolution.bm_dim,),
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
    """Dispatch between SDE solver backends.

    Args:
        source: Backend name (`"diffrax"` or `"em_scan"`).
        dynamics: Dynamical model with continuous-time state evolution.
        t0: Initial time.
        saveat_times: Times at which to return states.
        x0: Initial state.
        control_path_eval: Optional control evaluator `u(t)`.
        diffeqsolve_settings: Backend-specific solver settings.
        key: PRNG key for stochastic integration.
        tol_vbt: VirtualBrownianTree tolerance for the diffrax backend.

    Returns:
        Simulated state trajectory at `saveat_times`.
    """
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
