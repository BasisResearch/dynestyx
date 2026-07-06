"""Shared simulator helpers and base handler logic."""

import itertools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax import Array
from jaxtyping import Real
from numpyro.contrib.control_flow import scan as nscan

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.plate_utils import (
    _slice_array_for_plate_member,
    _slice_dist_for_plate_member,
)
from dynestyx.models import (
    Diffusion,
    DynamicalModel,
)
from dynestyx.observation_missingness import (
    ObservationLogProb,
)
from dynestyx.types import FunctionOfTime, SimulatedResult
from dynestyx.utils import (
    _diffusion_coefficient_is_plate_batched,
    _dist_has_plate_batch_dims,
    _get_val_or_None,
    _has_any_batched_plate_source,
    _is_opaque_plate_leaf,
    _leaf_is_plate_batched,
    _validate_site_sorting,
)


def _tile_times(times: Array, n_sim: int) -> Array:
    """Return times tiled to shape (n_sim, T)."""
    return jnp.broadcast_to(jnp.expand_dims(times, axis=0), (n_sim, len(times)))


def _ensure_trailing_dim(arr: Array) -> Array:
    """Ensure simulator outputs follow shape (n_sim, T, dim)."""
    return arr[..., jnp.newaxis] if arr.ndim == 2 else arr


def _merge_segments(
    arr_list: list[Array],
    seg_masks: list[Array],
    n_pred: int,
) -> Array:
    """Merge segment outputs into one array in predict-time order.

    Each segment contributes values only where its mask is True. Input arrays
    must already be shaped (n_sim, T_seg, dim).
    """
    first = arr_list[0]
    assert first.ndim == 3, (
        f"_merge_segments expects ndim==3 arrays (n_sim, T, D), got ndim={first.ndim} "
        f"with shape {first.shape}. Ensure _ensure_trailing_dim is applied before "
        "calling this function."
    )
    out = jnp.zeros((first.shape[0], n_pred, first.shape[2]), dtype=first.dtype)
    for arr, mask in zip(arr_list, seg_masks):
        cumsum = jnp.cumsum(mask)
        local_idx = jnp.where(mask, cumsum - 1, 0)
        gathered = arr[:, local_idx, :]
        mask_bc = jnp.expand_dims(jnp.expand_dims(mask, 0), -1)  # (1, T, 1)
        out = jnp.where(mask_bc, gathered, out)
    return out


def _simulated_result_to_dict(result: SimulatedResult) -> dict[str, Array]:
    """Convert a pure simulation result into deterministic-site payloads."""
    return {
        "times": jnp.asarray(result.times),
        "states": jnp.asarray(result.states),
        "observations": jnp.asarray(result.observations),
    }


_SIMULATOR_CONFIG_UNSET = object()


def _validate_no_config_and_direct_kwargs(
    *,
    simulator_config,
    config_name: str,
    direct_kwargs: dict[str, Any],
) -> None:
    """Reject ambiguous mixed config/direct simulator constructor usage."""
    if simulator_config is None:
        return

    provided_direct = sorted(
        name
        for name, value in direct_kwargs.items()
        if value is not _SIMULATOR_CONFIG_UNSET
    )
    if not provided_direct:
        return

    provided_str = ", ".join(provided_direct)
    raise ValueError(
        f"Received both {config_name} and direct simulator kwargs ({provided_str}). "
        f"Please provide either {config_name} or direct kwargs, not both."
    )


@contextmanager
def _suspend_numpyro_plate_frames():
    """Temporarily remove active numpyro.plate frames from the pyro stack.

    This is necessary so that `numpyro.sample` statements can be called within
    the simulator inside of a dsx.plate context."""
    stack = numpyro.primitives._PYRO_STACK
    original = list(stack)
    stack[:] = [f for f in original if not isinstance(f, numpyro.primitives.plate)]
    try:
        yield
    finally:
        stack[:] = original


def _slice_tree_for_plate_member(tree, plate_shapes: tuple[int, ...], plate_idx):
    """Slice plate-batched dynamics leaves for one simulator plate member.

    Shared leaves pass through unchanged; plate-batched leaves are selected by
    ``plate_idx``. Distribution parameters, including initial conditions, are
    sliced separately by ``_slice_dist_for_plate_member``.
    """

    def _slice_leaf(path, leaf):
        # Only constant-coefficient diffusions are opaque leaves (see
        # ``_is_opaque_plate_leaf``), so indexing the coefficient by ``plate_idx``
        # is well-defined; a callable coefficient is recursed into and its array
        # fields are sliced generically by the branch below.
        if isinstance(leaf, Diffusion):
            if _diffusion_coefficient_is_plate_batched(leaf, plate_shapes):
                return eqx.tree_at(
                    lambda d: d.coefficient, leaf, leaf.coefficient[plate_idx]
                )
            return leaf
        if _leaf_is_plate_batched(leaf, plate_shapes, path=path):
            return leaf[plate_idx]
        return leaf

    return jax.tree_util.tree_map_with_path(
        _slice_leaf,
        tree,
        is_leaf=_is_opaque_plate_leaf,
    )


class BaseSimulator(ObjectInterpretation, HandlesSelf):
    """Base class for generation-only simulator/unroller handlers.

    Interprets `dsx.sample(name, dynamics, predict_times=..., ...)` by unrolling
    `dynamics` into NumPyro sample sites for forward simulation on the requested
    prediction grid.

    When the simulator runs, it records the solved trajectories as deterministic
    sites (conventionally `"times"`, `"states"`, and `"observations"`).

    Notes:
        - Raw simulator handlers are generation-only and therefore require
          `predict_times`.
        - Observation-conditioned latent-state inference now belongs to
          `LatentPathBuilder` (explicit latent paths) or `Filter` / `Smoother`
          (marginalized inference).
        - Posterior rollout remains supported because `Filter` / `Smoother`
          consume `obs_times` / `obs_values` before forwarding rollout metadata
          to the simulator.
    """

    n_simulations: int = 1

    def _run_single_member_simulation(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        _obs_values_filled=None,
        _obs_mask=None,
        _obs_has_missing=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        filtered_times=None,
        filtered_dists=None,
        smoothed_times=None,
        smoothed_dists=None,
        _posterior_rollout_final_only: bool = False,
        **kwargs,
    ) -> dict[str, Array] | None:
        """Run simulator logic for one unbatched member and return trajectories."""
        use_smoothed_rollout = smoothed_times is not None or smoothed_dists is not None
        if use_smoothed_rollout and (
            filtered_times is not None or filtered_dists is not None
        ):
            raise ValueError(
                "Smoothed rollout metadata was provided alongside filtered rollout "
                "metadata. When smoothed_times or smoothed_dists is provided, "
                "filtered_times and filtered_dists must be None."
            )
        rollout_times = smoothed_times if use_smoothed_rollout else filtered_times
        rollout_dists = smoothed_dists if use_smoothed_rollout else filtered_dists
        rollout_label = "smoothed" if use_smoothed_rollout else "filtered"
        if (
            rollout_times is not None
            and rollout_dists is None
            and predict_times is not None
        ):
            raise ValueError(
                f"Rollout requested with {rollout_label}_times but missing {rollout_label}_dists. "
                "Plate-aware rollout requires posterior distributions from Filter/Smoother."
            )

        # Need times to simulate: predict_times or obs_times
        # For posterior rollout, need predict_times
        if predict_times is None:
            if obs_times is None or rollout_times is not None:
                return None

        posterior_rollout = rollout_times is not None and rollout_dists is not None

        if posterior_rollout:
            assert predict_times is not None
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
                seg_result = self._simulate(
                    seg_name,
                    dynamics_seg,
                    obs_times=None,
                    obs_values=None,
                    ctrl_times=ctrl_t_seg,
                    ctrl_values=ctrl_v_seg,
                    predict_times=predict_times,
                )
                results = {
                    "predicted_states": seg_result["states"],
                    "predicted_observations": seg_result["observations"],
                }
                n_sim_out = results["predicted_states"].shape[0]
                results["predicted_times"] = _tile_times(predict_times, n_sim_out)
                return results

            n_pred = len(predict_times)

            # Build segment ids on host once.
            # seg_id == -1 means "before first posterior time" (use model prior).
            pt_host = np.asarray(jax.device_get(predict_times))
            ft_host = np.asarray(jax.device_get(rollout_times))
            seg_ids_host = np.searchsorted(ft_host, pt_host, side="right") - 1

            seg_results = []
            seg_masks = []
            # Simulate one segment per present anchor (skip empty segments).
            for seg_id in [int(s) for s in np.unique(seg_ids_host)]:
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

                ctrl_t_seg, ctrl_v_seg = _ctrl_for_segment(sub_times)
                seg_results.append(
                    self._simulate(
                        seg_name,
                        dynamics_seg,
                        obs_times=None,
                        obs_values=None,
                        ctrl_times=ctrl_t_seg,
                        ctrl_values=ctrl_v_seg,
                        predict_times=sub_times,
                    )
                )
                seg_masks.append(mask_seg)

            # Scatter each segment's output into the global predict_times order.
            merge = lambda key: _merge_segments(
                [r[key] for r in seg_results], seg_masks, n_pred
            )
            results = {
                "predicted_states": merge("states"),
                "predicted_observations": merge("observations"),
            }
            n_sim_out = results["predicted_states"].shape[0]
            results["predicted_times"] = _tile_times(predict_times, n_sim_out)
            return results

        if self.n_simulations > 1 and obs_values is not None:
            raise ValueError(
                "n_simulations > 1 is only supported when obs_values is None "
                "(forward simulation only)"
            )
        return self._simulate(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            _obs_values_filled=_obs_values_filled,
            _obs_mask=_obs_mask,
            _obs_has_missing=_obs_has_missing,
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
        plate_shapes: tuple[int, ...],
        obs_times=None,
        obs_values=None,
        _obs_values_filled=None,
        _obs_mask=None,
        _obs_has_missing=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        filtered_times=None,
        filtered_dists=None,
        smoothed_times=None,
        smoothed_dists=None,
        _posterior_rollout_final_only: bool = False,
        **kwargs,
    ) -> dict[str, Array] | None:
        """Run simulator over all plate members and stack outputs.

        Plated simulation enumerates over all plate members and runs
        individual simulations. This is somewhat slower than vmapping,
        but maintains full compatibility with NumPyro's sample semantics."""
        if not _has_any_batched_plate_source(
            dynamics,
            plate_shapes,
            arrays=(
                obs_times,
                obs_values,
                ctrl_times,
                ctrl_values,
                predict_times,
                filtered_times,
                smoothed_times,
            ),
            dists=smoothed_dists if smoothed_dists is not None else filtered_dists,
        ):
            raise ValueError(
                "Plate simulator received plate_shapes but no plate-batched dynamics/data "
                "sources were found. At least one source must have leading dimensions "
                "matching plate_shapes."
            )

        plate_indices = list(itertools.product(*[range(s) for s in plate_shapes]))
        member_results: list[dict[str, Array]] = []

        for plate_idx in plate_indices:
            member_name = f"{name}_p{'_'.join(str(i) for i in plate_idx)}"

            # We begin by slicing the dynamics tree for each plate member.
            member_dynamics = _slice_tree_for_plate_member(
                dynamics, plate_shapes, plate_idx
            )

            # If initial conditions have plate dimensions, we also slice & apply them.
            if _dist_has_plate_batch_dims(dynamics.initial_condition, plate_shapes):
                member_initial_condition = _slice_dist_for_plate_member(
                    dynamics.initial_condition, plate_shapes, plate_idx
                )
                member_dynamics = eqx.tree_at(
                    lambda m: m.initial_condition,
                    member_dynamics,
                    member_initial_condition,
                    is_leaf=lambda x: x is None,
                )

            # We then slice each other source to find the member's times/values.
            member_obs_times = _slice_array_for_plate_member(
                obs_times, plate_shapes, plate_idx
            )
            member_obs_values = _slice_array_for_plate_member(
                obs_values, plate_shapes, plate_idx
            )
            member_obs_values_filled = _slice_array_for_plate_member(
                _obs_values_filled, plate_shapes, plate_idx
            )
            member_obs_mask = _slice_array_for_plate_member(
                _obs_mask, plate_shapes, plate_idx
            )
            member_ctrl_times = _slice_array_for_plate_member(
                ctrl_times, plate_shapes, plate_idx
            )
            member_ctrl_values = _slice_array_for_plate_member(
                ctrl_values, plate_shapes, plate_idx
            )
            member_predict_times = _slice_array_for_plate_member(
                predict_times, plate_shapes, plate_idx
            )
            member_filtered_times = _slice_array_for_plate_member(
                filtered_times, plate_shapes, plate_idx
            )
            member_smoothed_times = _slice_array_for_plate_member(
                smoothed_times, plate_shapes, plate_idx
            )

            # Same distribution slicing logic as above, but for prediction.
            member_filtered_dists = None
            if filtered_dists is not None:
                member_filtered_dists = [
                    _slice_dist_for_plate_member(d, plate_shapes, plate_idx)
                    for d in filtered_dists
                ]
            member_smoothed_dists = None
            if smoothed_dists is not None:
                member_smoothed_dists = [
                    _slice_dist_for_plate_member(d, plate_shapes, plate_idx)
                    for d in smoothed_dists
                ]

            # To perform inference, we need to suspend the active numpyro.plate frames
            # This is because the simulator has unguarded numpyro.sample statements inside,
            # which would otherwise create nested plate frames.
            with _suspend_numpyro_plate_frames():
                member_result = self._run_single_member_simulation(
                    member_name,
                    member_dynamics,
                    obs_times=member_obs_times,
                    obs_values=member_obs_values,
                    _obs_values_filled=member_obs_values_filled,
                    _obs_mask=member_obs_mask,
                    _obs_has_missing=_obs_has_missing,
                    ctrl_times=member_ctrl_times,
                    ctrl_values=member_ctrl_values,
                    predict_times=member_predict_times,
                    filtered_times=member_filtered_times,
                    filtered_dists=member_filtered_dists,
                    smoothed_times=member_smoothed_times,
                    smoothed_dists=member_smoothed_dists,
                    _posterior_rollout_final_only=_posterior_rollout_final_only,
                    **kwargs,
                )

            if member_result is not None:
                member_results.append(member_result)

        if not member_results:
            return None

        keys = member_results[0].keys()
        for result in member_results:
            if result.keys() != keys:
                raise ValueError(
                    "Plate simulator members returned inconsistent result keys."
                )

        stacked: dict[str, Array] = {}
        for key in keys:
            values = [r[key] for r in member_results]
            flat = jnp.stack(values, axis=0)
            stacked[key] = flat.reshape(*plate_shapes, *values[0].shape)
        return stacked

    @implements(_condition_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
        obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
        | Real[Array, "*obs_value_plate obs_time"]
        | None = None,
        _obs_values_filled: Array | None = None,
        _obs_mask: Array | None = None,
        _obs_has_missing: bool | None = None,
        ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
        ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
        | Real[Array, "*ctrl_value_plate ctrl_time"]
        | None = None,
        predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
        filtered_times=None,
        filtered_dists=None,
        smoothed_times=None,
        smoothed_dists=None,
        **kwargs,
    ) -> FunctionOfTime:
        posterior_rollout_final_only = kwargs.pop(
            "_posterior_rollout_final_only", False
        )
        raw_simulator_request = (
            filtered_times is None
            and filtered_dists is None
            and smoothed_times is None
            and smoothed_dists is None
        )
        if raw_simulator_request and (obs_times is not None or obs_values is not None):
            raise ValueError(
                "Simulator handlers are generation-only and no longer accept "
                "obs_times/obs_values directly. Use predict_times for forward "
                "simulation, LatentPathBuilder for explicit latent-path inference, "
                "or Filter/Smoother for marginalized inference."
            )
        if plate_shapes:
            results = self._run_plated_simulation(
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
                filtered_times=filtered_times,
                filtered_dists=filtered_dists,
                smoothed_times=smoothed_times,
                smoothed_dists=smoothed_dists,
                _posterior_rollout_final_only=posterior_rollout_final_only,
                **kwargs,
            )
        else:
            results = self._run_single_member_simulation(
                name,
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                _obs_values_filled=_obs_values_filled,
                _obs_mask=_obs_mask,
                _obs_has_missing=_obs_has_missing,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=predict_times,
                filtered_times=filtered_times,
                filtered_dists=filtered_dists,
                smoothed_times=smoothed_times,
                smoothed_dists=smoothed_dists,
                _posterior_rollout_final_only=posterior_rollout_final_only,
                **kwargs,
            )

        if results is not None:
            # Add the results from the simulator as deterministic sites
            for site_name, trajectory in results.items():
                numpyro.deterministic(f"{name}_{site_name}", trajectory)

        return fwd(
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
            **kwargs,
        )

    def _simulate(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        _obs_values_filled=None,
        _obs_mask=None,
        _obs_has_missing=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, Array]:
        """Unroll `dynamics` as a NumPyro forward simulator.

        Implementations are expected to:
        - require `predict_times` for raw generation / rollout,
        - sample latent states and observations on that grid,
        - and return arrays suitable for recording as deterministic sites.

        Args:
            dynamics: Dynamical model to simulate/unroll.
            obs_times: Internal rollout metadata from higher-level handlers.
                Raw simulator use should leave this as `None`.
            obs_values: Internal rollout metadata from higher-level handlers.
                Raw simulator use should leave this as `None`.
            ctrl_times: Optional control times.
            ctrl_values: Optional control values aligned to `ctrl_times`.
            predict_times: Prediction times at which to emit forward-simulation
                sites.
        Returns:
            dict[str, State]: Mapping from deterministic site names to
                trajectories. Conventionally includes `"times"`, `"states"`,
                and `"observations"`.
        """
        raise NotImplementedError()

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: Array,
        obs_times=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> SimulatedResult:
        """Run pure-JAX forward simulation without registering NumPyro sites."""
        raise NotImplementedError()


def _emit_observations(
    name: str,
    dynamics,
    states: Array,
    times: Array,
    obs_values: Array | None,
    control_path_eval: Callable[[Array], Array | None],
    key=None,
) -> Array:
    """Emit observations via numpyro.sample (conditioning) or dist.sample (vmap)."""
    ctrl = control_path_eval if control_path_eval is not None else (lambda t: None)
    T = len(times)

    if key is not None:
        obs_keys = jr.split(key, T)

        def _obs_step(t_idx):
            x_t = states[t_idx]
            t = times[t_idx]
            u_t = ctrl(t)
            obs_dist = dynamics.observation_model(x=x_t, u=u_t, t=t)
            return obs_dist.sample(obs_keys[t_idx])

        return jax.vmap(_obs_step)(jnp.arange(T))
    else:

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


def _apply_observation_log_prob(
    name: str,
    states: Array,
    times: Array,
    log_prob: ObservationLogProb,
    control_path_eval: Callable[[Array], Array | None],
) -> Array:
    """Apply observation log-probability terms and preserve NaNs in outputs."""
    ctrl = control_path_eval if control_path_eval is not None else (lambda t: None)
    T = len(times)

    def _step(carry, t_idx):
        x_t = states[t_idx]
        t = times[t_idx]
        u_t = ctrl(t)
        lp = log_prob.log_prob_step(x=x_t, u=u_t, t=t, t_idx=t_idx)
        numpyro.factor(f"{name}_y_{t_idx}_lp", lp)
        return carry, log_prob.observation_step(t_idx)

    _, observations = nscan(_step, None, jnp.arange(T))
    for t_idx in range(T):
        numpyro.deterministic(f"{name}_y_{t_idx}", observations[t_idx])
    return observations
