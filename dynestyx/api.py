"""Top-level pure-JAX dynestyx APIs."""

import jax.numpy as jnp
from jaxtyping import Array, Real

from dynestyx.handlers import _validate_and_prepare
from dynestyx.inference.checkers import _validate_inference_supported_model_classes
from dynestyx.inference.configs.simulator import SimulatorConfig
from dynestyx.inference.state_paths.reconstruct import reconstruct_state_path
from dynestyx.inference.state_paths.score import compute_state_path_log_prob
from dynestyx.models import DynamicalModel
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
)
from dynestyx.types import SimulatedResult
from dynestyx.utils import (
    _get_dynamics_with_t0,
    _validate_control_dim,
    _validate_controls,
    _validate_site_sorting,
)


def simulate(
    dynamics: DynamicalModel,
    *,
    rng_key,
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
    n_simulations: int = 1,
    simulator_config: SimulatorConfig | None = None,
) -> SimulatedResult:
    """Run pure-JAX forward simulation and return a :class:`SimulatedResult`.

    Unlike :func:`dynestyx.sample`, this entry point is for data generation only
    and does not register any NumPyro sample/deterministic sites. The simulator
    time grid is taken from ``predict_times`` when provided, else from
    ``obs_times``.
    """
    if obs_times is None and predict_times is None:
        raise ValueError("At least one of obs_times or predict_times must be provided")

    _validate_site_sorting(obs_times, name="obs_times")
    _validate_site_sorting(ctrl_times, name="ctrl_times")
    _validate_site_sorting(predict_times, name="predict_times")
    _validate_controls(obs_times, predict_times, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)

    dynamics_with_t0 = _get_dynamics_with_t0(dynamics, obs_times, predict_times)

    from dynestyx.simulation import Simulator

    simulator = Simulator(
        n_simulations=n_simulations,
        simulator_config=simulator_config,
    )
    return simulator.simulate(
        dynamics_with_t0,
        rng_key=rng_key,
        obs_times=obs_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
    )


def log_prob(
    dynamics: DynamicalModel,
    *,
    state_path_params,
    state_path_param_times,
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None = None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    missing_obs_values=None,
    missing_obs_metadata: MissingObservationMetadata | None = None,
    chunk_size: int | None = None,
    ode_diffeqsolve_settings=None,
):
    """Return the pure-JAX joint log density for a state path and observations.

    Parameters:
        dynamics: Dynamical model to score.
        state_path_params: Concrete path-parameter values supplied by the
            caller. Writing ``z = state_path_params`` and
            ``x = g(z) = state_path``, this function evaluates ``log p(x, y)``
            after reconstructing the path ``x``. For discrete / discretized
            models ``z`` is the full latent path in v1. For ODE models ``z``
            is the initial condition in v1.
        state_path_param_times: Times attached to ``state_path_params``.
        obs_times: Times at which observations are available.
        obs_values: Observation values at ``obs_times``.
        ctrl_times: Times at which controls are supplied.
        ctrl_values: Control values aligned to ``ctrl_times``.
        missing_observation_strategy: Strategy for handling missing
            observation coordinates. `"auto"` prefers exact marginalization when
            supported and otherwise falls back to explicit augmentation for
            continuous observation families.
        missing_obs_values: Explicit values for the missing observation
            coordinates when augmentation is active. In that case this function
            scores the augmented complete-data target
            ``log p(x, y_observed, y_missing | ...)`` rather than the
            marginalized observed-data target.
        missing_obs_metadata: Optional precomputed metadata defining the flat
            ordering of ``missing_obs_values`` for traced/JIT callers.
        chunk_size: Optional host-level chunk size for scoring loops.
        ode_diffeqsolve_settings: Optional Diffrax solve settings used when
            reconstructing deterministic continuous-time trajectories.
    """
    state_path_param_times = jnp.asarray(state_path_param_times)
    _validate_site_sorting(state_path_param_times, name="state_path_param_times")
    _validate_inference_supported_model_classes(dynamics)
    dynamics_with_t0, obs_values_filled, obs_mask, _obs_has_missing = (
        _validate_and_prepare(
            "log_prob",
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=state_path_param_times,
        )
    )

    _, state_path, state_path_times = reconstruct_state_path(
        dynamics_with_t0,
        state_path_params=state_path_params,
        state_path_param_times=state_path_param_times,
        obs_times=obs_times,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        ode_diffeqsolve_settings=ode_diffeqsolve_settings,
    )
    return compute_state_path_log_prob(
        dynamics_with_t0,
        state_path=state_path,
        state_path_times=state_path_times,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        missing_observation_strategy=missing_observation_strategy,
        missing_obs_values=missing_obs_values,
        missing_obs_metadata=missing_obs_metadata,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        chunk_size=chunk_size,
    )


__all__ = ["log_prob", "simulate"]
