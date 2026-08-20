"""Top-level pure-JAX API for simulation and scoring. Consider using as an alternative
to the NumPyro-based API if simulation and scoring are the only requirements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, PyTree, Real

from dynestyx.handlers import _validate_and_prepare
from dynestyx.inference.checkers import _validate_inference_supported_model_classes
from dynestyx.inference.configs.filter import BaseFilterConfig
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

if TYPE_CHECKING:
    from dynestyx.control.discrete_controller_simulators import PolicyCallable


def simulate(
    dynamics: DynamicalModel,
    *,
    rng_key: PRNGKeyArray,
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None = None,
    predict_times: Real[Array, " predict_time"] | None = None,
    n_simulations: int = 1,
    simulator_config: SimulatorConfig | None = None,
    control_policy: PolicyCallable | None = None,
    filter_config: BaseFilterConfig | None = None,
    use_true_state: bool = False,
    initial_policy_state: PyTree | None = None,
) -> SimulatedResult:
    """Simulate states and observations without registering NumPyro sites.

    The simulation runs on the grid specified by `predict_times`.

    Args:
        dynamics: Dynamical model to simulate.
        rng_key: JAX pseudorandom number generator key.
        ctrl_times: Times associated with `ctrl_values`. If controls are
            provided, these times must match `predict_times`.
        ctrl_values: Control values, or `None` for an uncontrolled model.
        predict_times: Times at which to simulate states and observations.
        n_simulations: Number of independent trajectories to simulate.
        simulator_config: ODE or SDE solver configuration. Its type must match
            the model's state evolution. Discrete-time models do not accept a
            simulator configuration.
        control_policy: Optional control policy (see
            `dynestyx.control.discrete_controller_simulators.PolicyCallable`).
            When given, controls are computed online in closed loop via
            [DiscreteControlLoopSimulator][dynestyx.control.discrete_controller_simulators.DiscreteControlLoopSimulator]
            instead of being drawn from the uncontrolled/`ctrl_values`
            transition -- `ctrl_times`/`ctrl_values` must not be passed
            together with `control_policy`, and `simulator_config` is not
            accepted either.
        filter_config: Filter configuration forwarded to
            `DiscreteControlLoopSimulator` when `control_policy` is given;
            ignored otherwise.
        use_true_state: Forwarded to `DiscreteControlLoopSimulator` when
            `control_policy` is given; ignored otherwise. When `True`,
            `control_policy` observes the true state directly instead of a
            filtered belief, and `filter_config` must be left `None` (see
            `DiscreteControlLoopSimulator`'s `use_true_state` attribute).
        initial_policy_state: Initial policy state $s_0$, forwarded to
            `DiscreteControlLoopSimulator` when `control_policy` is given;
            ignored otherwise. Defaults to `None` (a stateless policy) --
            `control_policy` is never introspected for an `initial_state()`
            method, so a stateful policy's initial state must always be
            passed explicitly here.

    Returns:
        SimulatedResult: Simulated times, initial states, state paths, and
            observations.

    Raises:
        ValueError: If `predict_times` is not provided, controls are incomplete
            or incompatible with the model, or the simulator configuration does
            not match the model.
        equinox.EquinoxRuntimeError: If a time array is not strictly
            increasing, `ctrl_times` does not match the required time grid, or
            `dynamics.t0` does not match the first prediction time.
    """
    if predict_times is None:
        raise ValueError("predict_times must be provided")

    _validate_site_sorting(ctrl_times, name="ctrl_times")
    _validate_site_sorting(predict_times, name="predict_times")
    _validate_controls(None, predict_times, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)

    dynamics_with_t0 = _get_dynamics_with_t0(dynamics, None, predict_times)

    from dynestyx.simulation import Simulator

    simulator = Simulator(
        n_simulations=n_simulations,
        simulator_config=simulator_config,
        control_policy=control_policy,
        filter_config=filter_config,
        use_true_state=use_true_state,
    )
    _, simulation_key = jr.split(rng_key)
    return simulator.simulate(
        dynamics_with_t0,
        rng_key=simulation_key,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
        initial_policy_state=initial_policy_state,
    )


def log_prob(
    dynamics: DynamicalModel,
    *,
    state_path_params: Real[Array, "state_path_param_time state_dim"]
    | Real[Array, " _"]
    | Real[Array, ""],
    state_path_param_times: Real[Array, " state_path_param_time"],
    obs_times: Real[Array, " obs_time"] | None = None,
    obs_values: Real[Array, "obs_time observation_dim"]
    | Real[Array, " obs_time"]
    | None = None,
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None = None,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    missing_obs_values: Real[Array, " n_missing_obs"]
    | Real[Array, " obs_time"]
    | Real[Array, "obs_time observation_dim"]
    | Real[Array, ""]
    | None = None,
    missing_obs_metadata: MissingObservationMetadata | None = None,
    chunk_size: int | None = 0,
    ode_diffeqsolve_settings: dict[str, Any] | None = None,
) -> Real[Array, "*log_prob_batch"]:
    """Evaluate the joint log density of a reconstructed state path.

    The function reconstructs a complete state path from
    `state_path_params`, then evaluates its initial, transition, and
    observation terms. If observations are omitted, it evaluates only the
    state-path density.

    Args:
        dynamics: Dynamical model to score.
        state_path_params: Values used to reconstruct the latent state path.
            For a discrete or discretized model, provide the complete path. For
            a deterministic ODE, provide its initial state.
        state_path_param_times: Strictly increasing times associated with
            `state_path_params`. A deterministic ODE expects one time.
        obs_times: Strictly increasing times associated with `obs_values`.
            Every observation time must occur in the reconstructed path.
        obs_values: Observation values, including any missing entries. Provide
            this argument together with `obs_times`.
        ctrl_times: Strictly increasing times associated with `ctrl_values`.
            When controls are provided, these times must match the union of
            `obs_times` and `state_path_param_times`.
        ctrl_values: Control values, or `None` for an uncontrolled model.
        missing_observation_strategy: Method used to handle missing
            observations. `"auto"` marginalizes supported observation
            distributions and otherwise uses augmentation for continuous
            distributions.
        missing_obs_values: Values used to complete missing observations when
            augmentation is active. Supply either a flat vector ordered by
            `missing_obs_metadata`, a scalar for one missing entry, or a dense
            array shaped like `obs_values`; observed entries in a dense array
            are ignored. In this case, the result includes the density of these
            values instead of marginalizing them.
        missing_obs_metadata: Positions, times, and component indices for
            `missing_obs_values`. Precompute this metadata before JIT-compiled
            augmentation when the missingness pattern cannot be inspected
            eagerly.
        chunk_size: Batch size passed to `jax.lax.map` while scoring transition
            and observation terms. The default, `0`, evaluates all terms with
            one `jax.vmap`. `None` maps one term at a time. A positive integer
            evaluates batches of that size with `jax.vmap`.
        ode_diffeqsolve_settings: Diffrax settings used to reconstruct a
            deterministic ODE path.

    Returns:
        Array: Joint log density, retaining any distribution batch axes.

    Raises:
        ValueError: If time, path, control, observation, or
            missing-observation inputs are inconsistent; if the model is not
            supported for scoring; or if a native SDE has not been discretized.
        equinox.EquinoxRuntimeError: If a time array is not strictly
            increasing, `dynamics.t0` does not match the earliest supplied
            time, or a required observation or control time is absent.
        NotImplementedError: If the selected missing-observation strategy is
            unsupported by the observation distribution.
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
