"""Pure simulation helpers and public simulation API."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Literal, cast

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jaxtyping import PRNGKeyArray, Real

from dynestyx.models import (
    DynamicalModel,
)
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.simulation_utils import _ensure_trailing_dim, _tile_times
from dynestyx.simulator_configs import SimulatorConfig
from dynestyx.simulators import (
    DiscreteTimeSimulator,
    ODESimulator,
    SDESimulator,
    build_simulator_for_dynamics,
)
from dynestyx.solvers import solve_ode, solve_sde
from dynestyx.types import as_scalar_time_array
from dynestyx.utils import (
    _build_control_path,
    _get_dynamics_with_t0,
    _validate_control_dim,
    _validate_controls,
    _validate_site_sorting,
)


@dataclasses.dataclass
class SimulationResult:
    """Pure simulation result returned by :func:`simulate`."""

    times: Array
    states: Array
    observations: Array


def _sample_observations(
    *,
    dynamics: DynamicalModel,
    states: Array,
    times: Array,
    control_path_eval: Callable[[Array], Array | None],
    key: Array,
) -> Array:
    """Sample one trajectory of observations from a latent state path."""
    obs_keys = jr.split(key, len(times))

    def _sample_one(t_idx):
        x_t = states[t_idx]
        t = times[t_idx]
        u_t = control_path_eval(t)
        return dynamics.observation_model(x=x_t, u=u_t, t=t).sample(obs_keys[t_idx])

    return jax.vmap(_sample_one)(jnp.arange(len(times)))


def _simulate_discrete(
    dynamics: DynamicalModel,
    *,
    predict_times: Array,
    ctrl_values: Array | None,
    n_simulations: int,
    rng_key: PRNGKeyArray,
) -> SimulationResult:
    state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)

    def _simulate_one(key: Array, x0):
        keys_t = jr.split(key, len(predict_times))

        def _step(x_prev, t_idx):
            t_now = predict_times[t_idx]
            t_next = predict_times[t_idx + 1]
            u_now = None if ctrl_values is None else ctrl_values[t_idx]
            u_next = None if ctrl_values is None else ctrl_values[t_idx + 1]
            trans_dist = state_transition(
                x=x_prev,
                u=u_now,
                t_now=t_now,
                t_next=t_next,
            )
            k_trans, k_obs = jr.split(keys_t[t_idx + 1], 2)
            x_t = trans_dist.sample(k_trans)
            y_t = dynamics.observation_model(x=x_t, u=u_next, t=t_next).sample(k_obs)
            return x_t, (x_t, y_t)

        u_0 = None if ctrl_values is None else ctrl_values[0]
        y_0 = dynamics.observation_model(x=x0, u=u_0, t=predict_times[0]).sample(
            keys_t[0]
        )
        _, (scan_states, scan_obs) = jax.lax.scan(
            _step,
            x0,
            jnp.arange(len(predict_times) - 1),
        )
        states = jnp.concatenate([jnp.expand_dims(x0, 0), scan_states], axis=0)
        observations = jnp.concatenate([jnp.expand_dims(y_0, 0), scan_obs], axis=0)
        return states, observations

    init_key, path_key = jr.split(rng_key)
    initial_state = dynamics.initial_condition.sample(
        init_key, sample_shape=(n_simulations,)
    )
    keys = jr.split(path_key, n_simulations)
    states, observations = jax.vmap(_simulate_one)(keys, initial_state)
    return SimulationResult(
        times=_tile_times(predict_times, n_simulations),
        states=_ensure_trailing_dim(states),
        observations=_ensure_trailing_dim(observations),
    )


def _simulate_ode(
    dynamics: DynamicalModel,
    *,
    predict_times: Array,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
    n_simulations: int,
    rng_key: PRNGKeyArray,
    solver: dfx.AbstractSolver,
    adjoint: dfx.AbstractAdjoint,
    stepsize_controller: dfx.AbstractStepSizeController,
    dt0: Real[Array, ""],
    max_steps: int,
) -> SimulationResult:
    if ctrl_times is not None and ctrl_values is not None:
        control_path = _build_control_path(ctrl_times, ctrl_values, predict_times)
        control_path_eval = lambda t: control_path.evaluate(t, left=False)
    else:
        control_path_eval = lambda t: None

    t0 = dynamics.t0 if dynamics.t0 is not None else predict_times[0]
    diffeqsolve_settings = {
        "solver": solver,
        "stepsize_controller": stepsize_controller,
        "adjoint": adjoint,
        "dt0": dt0,
        "max_steps": max_steps,
    }

    def _simulate_one(x0, obs_key):
        states = solve_ode(
            dynamics,
            t0,
            predict_times,
            x0,
            control_path_eval,
            diffeqsolve_settings,
        )
        observations = _sample_observations(
            dynamics=dynamics,
            states=states,
            times=predict_times,
            control_path_eval=control_path_eval,
            key=obs_key,
        )
        return states, observations

    init_key, obs_key = jr.split(rng_key)
    initial_state = dynamics.initial_condition.sample(
        init_key, sample_shape=(n_simulations,)
    )
    obs_keys = jr.split(obs_key, n_simulations)
    states, observations = jax.vmap(_simulate_one)(initial_state, obs_keys)
    return SimulationResult(
        times=_tile_times(predict_times, n_simulations),
        states=states,
        observations=observations,
    )


def _simulate_sde(
    dynamics: DynamicalModel,
    *,
    predict_times: Array,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
    n_simulations: int,
    rng_key: PRNGKeyArray,
    source: Literal["diffrax", "em_scan"],
    solver: dfx.AbstractSolver,
    adjoint: dfx.AbstractAdjoint,
    stepsize_controller: dfx.AbstractStepSizeController,
    dt0: Real[Array, ""],
    tol_vbt: Real[Array, ""] | None,
    max_steps: int | None,
) -> SimulationResult:
    if ctrl_times is not None and ctrl_values is not None:
        control_path = _build_control_path(ctrl_times, ctrl_values, predict_times)
        control_path_eval = lambda t: control_path.evaluate(t, left=False)
    else:
        control_path_eval = lambda t: None

    t0 = dynamics.t0 if dynamics.t0 is not None else predict_times[0]
    diffeqsolve_settings = {
        "solver": solver,
        "stepsize_controller": stepsize_controller,
        "adjoint": adjoint,
        "dt0": dt0,
        "max_steps": max_steps,
    }

    def _simulate_one(key: Array, x0):
        solve_key, obs_key = jr.split(key, 2)
        states = solve_sde(
            source=source,
            dynamics=dynamics,
            t0=t0,
            saveat_times=predict_times,
            x0=x0,
            control_path_eval=control_path_eval,
            diffeqsolve_settings=diffeqsolve_settings,
            key=solve_key,
            tol_vbt=tol_vbt,
        )
        observations = _sample_observations(
            dynamics=dynamics,
            states=states,
            times=predict_times,
            control_path_eval=control_path_eval,
            key=obs_key,
        )
        return states, observations

    init_key, path_key = jr.split(rng_key)
    initial_state = dynamics.initial_condition.sample(
        init_key, sample_shape=(n_simulations,)
    )
    keys = jr.split(path_key, n_simulations)
    states, observations = jax.vmap(_simulate_one)(keys, initial_state)
    return SimulationResult(
        times=_tile_times(predict_times, n_simulations),
        states=states,
        observations=observations,
    )


def simulate(
    dynamics: DynamicalModel,
    *,
    predict_times: Real[Array, "*predict_time_plate predict_time"],
    rng_key: PRNGKeyArray,
    n_simulations: int = 1,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    simulator_config: SimulatorConfig | None = None,
) -> SimulationResult:
    """Simulate a dynamical model without entering a NumPyro trace.

    `n_simulations` stays on the high-level API because it is commonly varied
    across all simulator types. Pass a `simulator_config` to choose less-common
    solver/backend settings explicitly. If `simulator_config` is omitted, the
    function auto-detects the model type and uses the same defaults as
    `Simulator()`, `ODESimulator()`, and `SDESimulator()`.
    """
    _validate_site_sorting(predict_times, name="predict_times")
    _validate_controls(None, predict_times, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)
    dynamics = _get_dynamics_with_t0(dynamics, None, predict_times)

    predict_times = jnp.asarray(predict_times)
    if predict_times.ndim != 1:
        raise ValueError(
            "simulate currently expects a single prediction grid with shape (time,)."
        )

    simulator = build_simulator_for_dynamics(
        dynamics,
        n_simulations=n_simulations,
        simulator_config=simulator_config,
    )

    if isinstance(simulator, DiscreteTimeSimulator):
        return _simulate_discrete(
            dynamics,
            predict_times=predict_times,
            ctrl_values=None if ctrl_values is None else jnp.asarray(ctrl_values),
            n_simulations=simulator.n_simulations,
            rng_key=rng_key,
        )

    if isinstance(simulator, ODESimulator):
        ode_settings = simulator.diffeqsolve_settings
        return _simulate_ode(
            dynamics,
            predict_times=predict_times,
            ctrl_times=None if ctrl_times is None else jnp.asarray(ctrl_times),
            ctrl_values=None if ctrl_values is None else jnp.asarray(ctrl_values),
            n_simulations=simulator.n_simulations,
            rng_key=rng_key,
            solver=ode_settings["solver"],
            adjoint=ode_settings["adjoint"],
            stepsize_controller=ode_settings["stepsize_controller"],
            dt0=as_scalar_time_array(
                ode_settings["dt0"],
                name="dt0",
                dtype=predict_times.dtype,
            ),
            max_steps=ode_settings["max_steps"],
        )

    if isinstance(simulator, SDESimulator):
        sde_settings = simulator.diffeqsolve_settings
        sde_dt0 = as_scalar_time_array(
            sde_settings["dt0"],
            name="dt0",
            dtype=predict_times.dtype,
        )
        tol_vbt_arr = None
        if simulator.source == "diffrax":
            tol_vbt_arr = as_scalar_time_array(
                sde_dt0 / 2 if simulator.tol_vbt is None else simulator.tol_vbt,
                name="tol_vbt",
                dtype=predict_times.dtype,
            )
        return _simulate_sde(
            dynamics,
            predict_times=predict_times,
            ctrl_times=None if ctrl_times is None else jnp.asarray(ctrl_times),
            ctrl_values=None if ctrl_values is None else jnp.asarray(ctrl_values),
            n_simulations=simulator.n_simulations,
            rng_key=rng_key,
            source=simulator.source,
            solver=sde_settings["solver"],
            adjoint=sde_settings["adjoint"],
            stepsize_controller=sde_settings["stepsize_controller"],
            dt0=sde_dt0,
            tol_vbt=tol_vbt_arr,
            max_steps=sde_settings["max_steps"],
        )

    raise ValueError(f"Unsupported simulator instance for simulate: {type(simulator)}")


__all__ = ["SimulationResult", "simulate"]
