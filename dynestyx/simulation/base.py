"""Shared simulator helpers and base handler logic."""

import dataclasses
import itertools
from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jaxtyping import Array, Bool, PRNGKeyArray, Real
from numpyro.contrib.control_flow import scan as nscan

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.utils.plate_utils import (
    _slice_array_for_plate_member,
    _slice_dist_for_plate_member,
    _slice_dynamics_for_plate_member,
)
from dynestyx.models import DynamicalModel
from dynestyx.simulation.utils import (
    _merge_segments,
    _register_simulated_result_sites,
    _sample_observation_path,
    _stack_simulated_results,
    _tile_times,
)
from dynestyx.types import (
    ConditionedResult,
    SimulatedResult,
    chain_numpyro_site_registrations,
)
from dynestyx.utils import (
    _get_val_or_None,
    _has_any_batched_plate_source,
    _validate_site_sorting,
)


def _slice_rollout_result_for_plate_member(
    result: ConditionedResult | None,
    plate_shapes: tuple[int, ...],
    plate_idx: tuple[int, ...],
) -> ConditionedResult | None:
    """Extract the rollout-bearing fields for one plate member."""
    if result is None:
        return None

    member_times = _slice_array_for_plate_member(result.times, plate_shapes, plate_idx)
    member_dists = None
    if result.dists is not None:
        member_dists = [
            _slice_dist_for_plate_member(dist, plate_shapes, plate_idx)
            for dist in result.dists
        ]

    return ConditionedResult(times=member_times, dists=member_dists)


class BaseSimulator(ObjectInterpretation, HandlesSelf):
    """Base class for generation-only simulator handlers.

    Interprets `dsx.sample(name, dynamics, predict_times=..., ...)` by running a
    pure-JAX forward simulation on the requested prediction grid, then
    registering the realized simulator outputs as deferred NumPyro sites only
    when the NumPyro-style API is used.

    When the simulator runs, it records the solved trajectories as deterministic
    sites (conventionally `"x_0"`, `"times"`, `"states"`, and
    `"observations"`).

    Notes:
        - Raw simulator handlers are generation-only and therefore require
          `predict_times`.
        - Observation-conditioned latent-state inference now belongs to
          `LatentPathBuilder` (explicit latent paths) or `Filter` / `Smoother`
          (marginalized inference).
        - Posterior rollout remains supported because `Filter` / `Smoother`
          condition on `obs_times` / `obs_values` before forwarding their result
          and the unchanged observation inputs to the simulator.
    """

    n_simulations: int = 1

    def __init__(self, *, n_simulations: int = 1) -> None:
        if n_simulations < 1:
            raise ValueError(
                "n_simulations must be greater than or equal to 1, "
                f"got {n_simulations}."
            )
        self.n_simulations = n_simulations

    def _validate_plate_support(self) -> None:
        """Validate that this simulator supports plated execution."""

    def _run_single_member_simulation(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray | None = None,
        ctrl_times: Real[Array, " ctrl_time"] | None = None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None = None,
        predict_times: Real[Array, " predict_time"] | None = None,
        filtered_result: ConditionedResult | None = None,
        smoothed_result: ConditionedResult | None = None,
        _posterior_rollout_final_only: bool = False,
        **kwargs,
    ) -> SimulatedResult | None:
        """Run simulator logic for one unbatched member and return trajectories."""
        if filtered_result is not None and smoothed_result is not None:
            raise ValueError(
                "Both filtered_result and smoothed_result were provided for posterior "
                "rollout. Provide exactly one inference result."
            )

        rollout_result = (
            smoothed_result if smoothed_result is not None else filtered_result
        )
        rollout_times = None if rollout_result is None else rollout_result.times
        rollout_dists = None if rollout_result is None else rollout_result.dists
        rollout_label = "smoothed" if smoothed_result is not None else "filtered"
        if (
            rollout_times is not None
            and rollout_dists is None
            and predict_times is not None
        ):
            raise ValueError(
                f"Rollout requested with {rollout_label} result times but missing "
                "posterior distributions. Plate-aware rollout requires distributions "
                "from Filter/Smoother."
            )
        if rollout_dists is not None and rollout_times is None:
            raise ValueError(
                f"Rollout requested with {rollout_label} result distributions but "
                "missing times."
            )

        if predict_times is None:
            return None

        posterior_rollout = rollout_times is not None and rollout_dists is not None

        if posterior_rollout:
            assert predict_times is not None
            if rollout_times.shape[-1] != len(rollout_dists):
                raise ValueError(
                    f"The {rollout_label} rollout result must provide one distribution "
                    "per time point: result.dists[i] corresponds to "
                    "result.times[..., i]. Got "
                    f"{rollout_times.shape[-1]} times and {len(rollout_dists)} "
                    "distributions."
                )
            if rng_key is None:
                raise ValueError("PRNG key required for simulator rollout.")
            _validate_site_sorting(rollout_times, name=f"{rollout_label}_times")

            def _ctrl_for_segment(sub_times):
                if ctrl_times is None or ctrl_values is None:
                    return None, None
                inds = jnp.searchsorted(ctrl_times, sub_times, side="left")
                return sub_times, ctrl_values[inds]

            def _dynamics_for_segment(seg_id: int):
                if seg_id < 0:
                    return dynamics, f"{name}_0"

                posterior_time = rollout_times[seg_id]
                posterior_dist = rollout_dists[seg_id]
                dynamics_with_posterior_time = eqx.tree_at(
                    lambda m: m.t0,
                    dynamics,
                    posterior_time,
                    is_leaf=lambda x: x is None,
                )
                dynamics_seg = eqx.tree_at(
                    lambda m: m.initial_condition,
                    dynamics_with_posterior_time,
                    posterior_dist,
                    is_leaf=lambda x: x is None,
                )
                return dynamics_seg, f"{name}_{seg_id + 1}"

            if _posterior_rollout_final_only:
                dynamics_seg, seg_name = _dynamics_for_segment(0)
                ctrl_t_seg, ctrl_v_seg = _ctrl_for_segment(predict_times)
                seg_result = self.simulate(
                    dynamics_seg,
                    rng_key=rng_key,
                    ctrl_times=ctrl_t_seg,
                    ctrl_values=ctrl_v_seg,
                    predict_times=predict_times,
                )
                assert seg_result.states is not None
                assert seg_result.observations is not None
                predicted_states = seg_result.states
                return SimulatedResult(
                    predicted_states=predicted_states,
                    predicted_observations=seg_result.observations,
                    predicted_times=_tile_times(
                        predict_times, predicted_states.shape[0]
                    ),
                    _register_numpyro_sites=lambda _site_name: (
                        _register_simulated_result_sites(
                            SimulatedResult(x_0=seg_result.x_0),
                            site_name=seg_name,
                        )
                    ),
                )

            n_pred = len(predict_times)

            # Build segment ids on host once.
            # seg_id == -1 means "before first posterior time" (use model prior).
            pt_host = np.asarray(jax.device_get(predict_times))
            ft_host = np.asarray(jax.device_get(rollout_times))
            seg_ids_host = np.searchsorted(ft_host, pt_host, side="right") - 1

            seg_results: list[SimulatedResult] = []
            seg_masks: list[Bool[Array, " predict_time"]] = []
            seg_names: list[str] = []
            nonempty_seg_ids = [int(s) for s in np.unique(seg_ids_host)]
            seg_keys = jr.split(rng_key, len(nonempty_seg_ids))
            # Simulate one segment per present anchor (skip empty segments).
            for seg_idx, seg_id in enumerate(nonempty_seg_ids):
                # mask_host[i] = True iff predict_times[i] belongs to this segment id.
                # This is the global-to-segment membership mask over the full prediction grid.
                mask_host = seg_ids_host == seg_id
                # Some segment ids may not own any prediction times. np.any here is host-side,
                # avoids traced bool conversion, and lets us skip empty segment solves.
                if not np.any(mask_host):
                    continue

                # Keep the same membership mask as a JAX array for scatter/merge later.
                mask_seg = jnp.asarray(mask_host)
                # Extract just this segment's prediction times (variable-length sub-grid).
                sub_times = jnp.asarray(pt_host[mask_host], dtype=predict_times.dtype)
                dynamics_seg, seg_name = _dynamics_for_segment(seg_id)
                seg_names.append(seg_name)

                ctrl_t_seg, ctrl_v_seg = _ctrl_for_segment(sub_times)
                seg_results.append(
                    self.simulate(
                        dynamics_seg,
                        rng_key=seg_keys[seg_idx],
                        ctrl_times=ctrl_t_seg,
                        ctrl_values=ctrl_v_seg,
                        predict_times=sub_times,
                    )
                )
                seg_masks.append(mask_seg)

            # Scatter each segment's output into the global predict_times order.
            def _merge_attr(
                attr: str,
            ) -> Real[Array, "n_simulations predict_time dim"]:
                return _merge_segments(
                    [cast(Array, getattr(result, attr)) for result in seg_results],
                    seg_masks,
                    n_pred,
                )

            predicted_states = _merge_attr("states")
            # The outer handler automatically registers vector fields after segments
            # (and any plate members) are aggregated. Preserve only unique,
            # segment-level metadata such as each realized x_0 in this callback.
            return SimulatedResult(
                predicted_states=predicted_states,
                predicted_observations=_merge_attr("observations"),
                predicted_times=_tile_times(predict_times, predicted_states.shape[0]),
                _register_numpyro_sites=chain_numpyro_site_registrations(
                    *(
                        (
                            lambda _site_name, seg_name=seg_name, seg_result=seg_result: (
                                _register_simulated_result_sites(
                                    SimulatedResult(x_0=seg_result.x_0),
                                    site_name=seg_name,
                                )
                            )
                        )
                        for seg_name, seg_result in zip(
                            seg_names, seg_results, strict=True
                        )
                    )
                ),
            )

        if rng_key is None:
            raise ValueError("PRNG key required for simulation.")
        return self.simulate(
            dynamics,
            rng_key=rng_key,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            **kwargs,
        )

    def _run_plated_simulation(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray | None = None,
        plate_shapes: tuple[int, ...],
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
        filtered_result: ConditionedResult | None = None,
        smoothed_result: ConditionedResult | None = None,
        _posterior_rollout_final_only: bool = False,
        **kwargs,
    ) -> SimulatedResult | None:
        """Run simulator over all plate members and stack outputs.

        Plated simulation enumerates over all plate members and runs
        individual simulations. This is somewhat slower than vmapping,
        but maintains full compatibility with NumPyro's sample semantics."""
        self._validate_plate_support()
        if predict_times is None:
            return None

        if not _has_any_batched_plate_source(
            dynamics,
            plate_shapes,
            arrays=(
                ctrl_times,
                ctrl_values,
                predict_times,
                None if filtered_result is None else filtered_result.times,
                None if smoothed_result is None else smoothed_result.times,
            ),
            dists=(
                smoothed_result.dists
                if smoothed_result is not None
                else None
                if filtered_result is None
                else filtered_result.dists
            ),
        ):
            raise ValueError(
                "Plate simulator received plate_shapes but no plate-batched dynamics/data "
                "sources were found. At least one source must have leading dimensions "
                "matching plate_shapes."
            )

        plate_indices = list(itertools.product(*[range(s) for s in plate_shapes]))
        member_results: list[SimulatedResult] = []
        member_keys = None if rng_key is None else jr.split(rng_key, len(plate_indices))

        for member_idx, plate_idx in enumerate(plate_indices):
            member_name = f"{name}_p{'_'.join(str(i) for i in plate_idx)}"

            member_dynamics = _slice_dynamics_for_plate_member(
                dynamics, plate_shapes, plate_idx
            )

            # We then slice each other source to find the member's times/values.
            member_ctrl_times = _slice_array_for_plate_member(
                ctrl_times, plate_shapes, plate_idx
            )
            member_ctrl_values = _slice_array_for_plate_member(
                ctrl_values, plate_shapes, plate_idx
            )
            member_predict_times = _slice_array_for_plate_member(
                predict_times, plate_shapes, plate_idx
            )
            member_filtered_result = _slice_rollout_result_for_plate_member(
                filtered_result, plate_shapes, plate_idx
            )
            member_smoothed_result = _slice_rollout_result_for_plate_member(
                smoothed_result, plate_shapes, plate_idx
            )

            member_result = self._run_single_member_simulation(
                member_name,
                member_dynamics,
                rng_key=(None if member_keys is None else member_keys[member_idx]),
                ctrl_times=member_ctrl_times,
                ctrl_values=member_ctrl_values,
                predict_times=member_predict_times,
                filtered_result=member_filtered_result,
                smoothed_result=member_smoothed_result,
                _posterior_rollout_final_only=_posterior_rollout_final_only,
                **kwargs,
            )

            if member_result is not None:
                member_results.append(member_result)

        if not member_results:
            return None

        return _stack_simulated_results(member_results, plate_shapes=plate_shapes)

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes: tuple[int, ...] = (),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_values_filled: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_mask: Bool[Array, "*obs_value_plate obs_time observation_dim"]
        | Bool[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_has_missing: bool | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
        filtered_result: ConditionedResult | None = None,
        smoothed_result: ConditionedResult | None = None,
        **kwargs,
    ) -> object:
        posterior_rollout_final_only = kwargs.pop(
            "_posterior_rollout_final_only", False
        )
        has_observations = obs_times is not None or obs_values is not None
        has_conditioned_result = (
            filtered_result is not None or smoothed_result is not None
        )
        if has_observations and not has_conditioned_result:
            raise ValueError(
                "Simulator handlers are generation-only and do not condition "
                "directly on obs_times/obs_values. Place Simulator outside "
                "Filter/Smoother for posterior rollout, use predict_times for prior "
                "simulation, or use LatentPathBuilder for explicit latent-path "
                "inference."
            )
        need_simulation = predict_times is not None
        simulation_key = None
        if need_simulation:
            simulation_key = numpyro.prng_key()
            if simulation_key is None:
                raise ValueError("PRNG key required for simulation.")
        if plate_shapes:
            results = self._run_plated_simulation(
                name,
                dynamics,
                rng_key=simulation_key,
                plate_shapes=plate_shapes,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=predict_times,
                filtered_result=filtered_result,
                smoothed_result=smoothed_result,
                _posterior_rollout_final_only=posterior_rollout_final_only,
                **kwargs,
            )
        else:
            results = self._run_single_member_simulation(
                name,
                dynamics,
                rng_key=simulation_key,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=predict_times,
                filtered_result=filtered_result,
                smoothed_result=smoothed_result,
                _posterior_rollout_final_only=posterior_rollout_final_only,
                **kwargs,
            )

        downstream_result = fwd(
            name,
            dynamics,
            plate_shapes=plate_shapes,
            obs_times=obs_times,
            obs_values=obs_values,
            _obs_values_filled=_obs_values_filled,
            _obs_mask=_obs_mask,
            _obs_has_missing=_obs_has_missing,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            filtered_result=filtered_result,
            smoothed_result=smoothed_result,
            **kwargs,
        )

        if results is None:
            return downstream_result

        def _register_self(site_name: str) -> None:
            _register_simulated_result_sites(results, site_name=site_name)

        downstream_register = getattr(
            downstream_result, "_register_numpyro_sites", None
        )
        return dataclasses.replace(
            results,
            _register_numpyro_sites=chain_numpyro_site_registrations(
                _register_self,
                results._register_numpyro_sites,
                downstream_register,
            ),
        )

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: PRNGKeyArray,
        ctrl_times: Real[Array, " ctrl_time"] | None = None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None = None,
        predict_times: Real[Array, " predict_time"] | None = None,
        **kwargs,
    ) -> SimulatedResult:
        """Run pure-JAX forward simulation without registering NumPyro sites."""
        raise NotImplementedError()

    def _emit_observations(
        self,
        name: str,
        dynamics: DynamicalModel,
        states: Real[Array, "time state_dim"] | Real[Array, " time"],
        times: Real[Array, " time"],
        obs_values: Real[Array, "time observation_dim"] | Real[Array, " time"] | None,
        control_path_eval: Callable[[Real[Array, ""]], Real[Array, "..."] | None]
        | None,
        key: PRNGKeyArray | None = None,
    ) -> Real[Array, "time observation_dim"] | Real[Array, " time"]:
        """Emit observations in pure-JAX or NumPyro mode."""
        ctrl = control_path_eval if control_path_eval is not None else (lambda t: None)
        T = len(times)

        if key is not None:
            return _sample_observation_path(
                dynamics,
                states=states,
                times=times,
                rng_key=key,
                control_path_eval=control_path_eval,
            )

        def _step(carry, t_idx):
            x_t = states[t_idx]
            t = times[t_idx]
            u_t = ctrl(t)
            obs_t = _get_val_or_None(obs_values, t_idx)
            y_t = numpyro.sample(
                f"{name}_y_{t_idx}",
                dynamics.observation_model(x=x_t, u=u_t, t=t),
                obs=obs_t,
            )
            return carry, y_t

        _, observations = nscan(_step, None, jnp.arange(T))
        return observations
