"""NumPyro-aware simulators/unrollers for dynamical models."""

import dataclasses
import itertools
from collections.abc import Callable
from contextlib import contextmanager
from typing import Literal, cast

import diffrax as dfx
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

from dynestyx.handlers import HandlesSelf, _condition_intp
from dynestyx.inference.plate_utils import (
    _slice_array_for_plate_member,
    _slice_dist_for_plate_member,
)
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    Diffusion,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.simulation_utils import _ensure_trailing_dim, _tile_times
from dynestyx.simulator_configs import (
    ODESimulatorConfig,
    SDESimulatorConfig,
    SimulatorConfig,
)
from dynestyx.solvers import solve_ode, solve_sde
from dynestyx.types import FunctionOfTime, as_scalar_time_array
from dynestyx.utils import (
    _build_control_path,
    _diffusion_coefficient_is_plate_batched,
    _dist_has_plate_batch_dims,
    _get_val_or_None,
    _has_any_batched_plate_source,
    _is_opaque_plate_leaf,
    _leaf_is_plate_batched,
    _validate_site_sorting,
)


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
    """Base class for generation-only simulator handlers.

    Interprets `dsx.sample(name, dynamics, predict_times=...)` by unrolling
    `dynamics` into NumPyro sample sites on the provided prediction grid.

    When the simulator runs, it records the solved trajectories as deterministic
    sites (conventionally `"times"`, `"states"`, and `"observations"`).

    Notes:
        - Simulators are generation-only and reject observation conditioning.
        - Posterior rollout from Filter/Smoother remains supported via
          `filtered_*` / `smoothed_*` metadata and `predict_times`.
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

        if predict_times is None:
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
        if obs_times is not None or obs_values is not None:
            raise ValueError(
                "Simulator handlers are generation-only and do not accept "
                "obs_times or obs_values. Use LatentStateBuilder for explicit "
                "latent-state inference, or Filter/Smoother for marginalized inference."
            )

        posterior_rollout_final_only = kwargs.pop(
            "_posterior_rollout_final_only", False
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
        """Generate trajectories from `dynamics`.

        Implementations are expected to:
        - use `predict_times` (or an internal rollout grid) as the simulation grid,
        - draw only the initial condition via `numpyro.sample(...)`,
        - sample all downstream states/observations in pure JAX,
        - and return arrays suitable for recording as deterministic sites.

        Args:
            dynamics: Dynamical model to simulate/unroll.
            obs_times: Reserved for compatibility with the shared handler
                signature; generation-only simulators reject it.
            obs_values: Reserved for compatibility with the shared handler
                signature; generation-only simulators reject it.
            ctrl_times: Optional control times.
            ctrl_values: Optional control values aligned to `ctrl_times`.
            predict_times: Prediction times at which trajectories are generated.
        Returns:
            dict[str, State]: Mapping from deterministic site names to
                trajectories. Conventionally includes `"times"`, `"states"`,
                and `"observations"`.
        """
        raise NotImplementedError()


def _emit_observations(
    dynamics,
    states: Array,
    times: Array,
    control_path_eval: Callable[[Array], Array | None],
    key: Array,
) -> Array:
    """Sample one trajectory of observations in pure JAX."""
    ctrl = control_path_eval if control_path_eval is not None else (lambda t: None)
    T = len(times)
    obs_keys = jr.split(key, T)

    def _obs_step(t_idx):
        x_t = states[t_idx]
        t = times[t_idx]
        u_t = ctrl(t)
        obs_dist = dynamics.observation_model(x=x_t, u=u_t, t=t)
        return obs_dist.sample(obs_keys[t_idx])

    return jax.vmap(_obs_step)(jnp.arange(T))


class SDESimulator(BaseSimulator):
    """Simulator for continuous-time stochastic dynamics (SDEs).

    This simulator integrates a `ContinuousTimeStateEvolution` with nonzero diffusion
    using Diffrax and a `VirtualBrownianTree` (see the Diffrax docs on
    [Brownian controls](https://docs.kidger.site/diffrax/api/brownian/)). It keeps
    only the initial condition as a NumPyro sample site and generates the rest of
    the rollout in pure JAX.

    Controls:
        If `ctrl_times` / `ctrl_values` are provided at the `dsx.sample(...)` site,
        controls are interpolated with a right-continuous rectilinear rule
        (`left=False`), i.e., the control at time `t_k` is `ctrl_values[k]`.

    Deterministic outputs:
        When run, the simulator records `"times"`, `"states"`, and `"observations"`
        as `numpyro.deterministic(...)` sites.

    Important:
        - This is intended for **simulation / predictive checks** inside NumPyro.
        - Conditioning on `obs_values` with an SDE unroller typically yields a
          very high-dimensional latent path and is usually a **poor inference
          strategy** for parameters. Prefer filtering (`Filter` with
          `ContinuousTime*Config`) or particle methods instead.

    Tip for speed:
        - Use `source="em_scan"` if you are happy with a simple Euler-Maruyama forward simulation
          (10–20x faster than Diffrax's implementation; see
          [Diffrax Issue #517](https://github.com/patrick-kidger/diffrax/issues/517)).
        - Use `source="diffrax"` if you want greater flexibility in the solver and step-size control.
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver = dfx.Heun(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        dt0: float | int | Array = 1e-4,
        tol_vbt: float | int | Array | None = None,
        max_steps: int | None = None,
        n_simulations: int = 1,
        source: Literal["diffrax", "em_scan"] = "em_scan",
    ):
        """Configure SDE integration settings.

        Args:
            solver: Diffrax solver for the SDE (e.g., [`dfx.Heun`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/)).
                For solver guidance, see [How to choose a solver](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/).
            stepsize_controller: Diffrax step-size controller. Use
                [`dfx.ConstantStepSize`](https://docs.kidger.site/diffrax/api/stepsize_controller/)
                for fixed-step simulation, or an adaptive controller for error-controlled stepping.
            adjoint: Diffrax adjoint strategy used for differentiation through the
                solver (relevant when used under gradient-based inference). See
                [Adjoints](https://docs.kidger.site/diffrax/api/adjoints/).
            dt0: Initial step size (float or JAX array) passed to
                [`diffrax.diffeqsolve`](https://docs.kidger.site/diffrax/api/diffeqsolve/).
            tol_vbt: Tolerance parameter for
                [`diffrax.VirtualBrownianTree`](https://docs.kidger.site/diffrax/api/brownian/). If None,
                defaults to `dt0 / 2`. For statistically correct simulation, this
                must be smaller than `dt0`.
            max_steps: Optional hard cap on solver steps.
            n_simulations: Number of independent trajectory simulations. When > 1,
                states and observations have an extra leading dimension (n_simulations, T, ...).
            source: SDE backend to use. `"diffrax"` uses Diffrax + Brownian tree.
                `"em_scan"` uses a custom fixed-step Euler-Maruyama `lax.scan`
                that advances at every `dt0` tick and also lands exactly on all
                requested solve times. Default is `"em_scan"` for speed.

        Notes:
            - `VirtualBrownianTree` draws randomness via `numpyro.prng_key()`, so
              `SDESimulator` must be executed inside a seeded NumPyro context.
        """
        dt0_arr = as_scalar_time_array(dt0, name="dt0")
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0_arr,
            "max_steps": max_steps,
        }
        self.n_simulations = n_simulations
        self.source = source
        if self.source not in {"diffrax", "em_scan"}:
            raise ValueError(
                "SDESimulator source must be one of {'diffrax', 'em_scan'}, "
                f"got source={self.source!r}."
            )

        self.tol_vbt: Real[Array, ""] | None
        if self.source == "diffrax":
            if tol_vbt is None:
                self.tol_vbt = dt0_arr / 2.0
            else:
                self.tol_vbt = as_scalar_time_array(tol_vbt, name="tol_vbt")

            assert self.tol_vbt < dt0_arr, (
                "tol_vbt must be smaller than dt0 for statistically correct simulation."
            )
        else:
            # tol_vbt is only used by the diffrax backend.
            self.tol_vbt = None

    def _simulate(
        self,
        name: str,
        dynamics,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, Array]:
        """
        Simulate a continuous-time SDE after drawing only the initial state.

        This method:
        - samples the initial latent state as `numpyro.sample("x_0", ...)`,
        - integrates the SDE to all `obs_times` using Diffrax,
        - samples observations in pure JAX on the same grid,
        - and returns trajectories for deterministic recording.

        To handle controls, we use a rectilinear interpolation that is right-continuous,
        i.e., if ctrl_times = [0.0, 1.0, 2.0] and ctrl_values = [0.0, 1.0, 2.0],
        then the control at time 1.0 is the value at time 1.0.

        Args:
            dynamics: A `DynamicalModel` whose `state_evolution` is a
                `ContinuousTimeStateEvolution` with a non-None diffusion
                and inferred `bm_dim` (set during `DynamicalModel` construction).
            obs_times: Times at which to save the latent state and emit observations.
                Required.
            obs_values: Reserved for compatibility with the shared handler
                signature; generation-only simulators reject it.
            ctrl_times: Optional control times.
            ctrl_values: Optional control values aligned to `ctrl_times`.
            predict_times: Prediction times at which trajectories are generated.
        Returns:
            dict[str, State]: Dictionary with `"times"`, `"states"`, and
                `"observations"` trajectories.

        Warning:
            Conditioning on `obs_values` here is generally **not** a good way to do
            parameter inference for SDEs, because it introduces an explicit, high-
            dimensional latent path. Prefer filtering (`Filter`) or particle methods.
        """
        if not isinstance(
            dynamics.state_evolution, StochasticContinuousTimeStateEvolution
        ):
            raise NotImplementedError(
                "SDESimulator only works with StochasticContinuousTimeStateEvolution, got "
                f"{type(dynamics.state_evolution)}"
            )

        if obs_times is not None:
            raise ValueError(
                "obs_times must not be provided to an SDESimulator; it cannot be used for inference. \
                Please use a filter, or discretize the SDE and use a DiscreteTimeSimulator. \
                A natural example forthcoming (i.e., to be implemented) is the SimulatedLikelihoodDiscretizer."
            )

        if obs_values is not None:
            raise ValueError(
                "obs_values conditioning is not supported for SDESimulator. "
                "Use Filter for inference with SDEs."
            )

        times = predict_times
        if times is None:
            raise ValueError("predict_times must be provided for SDESimulator.")
        n_sim = self.n_simulations

        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, times)
            control_path_eval: Callable[[Array], Array | None] = lambda t: (
                control_path.evaluate(t, left=False)
            )
        else:
            control_path_eval = lambda t: None

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]

        def _sim_one_trajectory(key: Array, x0: Array) -> tuple[Array, Array]:
            """Simulate one SDE trajectory and its observations."""
            k_solve, k_obs = jr.split(key, 2)
            states_sol = solve_sde(
                source=self.source,
                dynamics=dynamics,
                t0=t0,
                saveat_times=times,
                x0=x0,
                control_path_eval=control_path_eval,
                diffeqsolve_settings=self.diffeqsolve_settings,
                key=k_solve,
                tol_vbt=self.tol_vbt,
            )
            emissions = _emit_observations(
                dynamics, states_sol, times, control_path_eval, k_obs
            )
            return states_sol, emissions

        prng_key = numpyro.prng_key()
        if prng_key is None:
            raise ValueError("PRNG key required for simulation")
        with numpyro.plate(f"{name}_n_simulations", n_sim):
            initial_state = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        keys = jr.split(prng_key, n_sim)
        states, emissions = jax.vmap(_sim_one_trajectory)(
            keys, jnp.asarray(initial_state)
        )
        return {
            "times": _tile_times(times, n_sim),
            "states": states,
            "observations": emissions,
        }


@dataclasses.dataclass
class DiscreteTimeSimulator(BaseSimulator):
    """Generation-only simulator for discrete-time dynamical models."""

    n_simulations: int = 1

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
        """Sample a discrete-time trajectory after drawing only the initial state."""
        del _obs_values_filled, _obs_mask, _obs_has_missing, kwargs
        times = obs_times if obs_times is not None else predict_times
        if times is None:
            raise ValueError("obs_times or predict_times must be provided")
        if obs_values is not None:
            raise ValueError(
                "DiscreteTimeSimulator is generation-only. Use predict_times for "
                "forward simulation, LatentStateBuilder for explicit latent-state "
                "inference, or Filter for marginalized inference."
            )

        T = len(times)
        if T < 1:
            raise ValueError("obs_times must contain at least one timepoint")

        n_sim = self.n_simulations

        state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)
        with numpyro.plate(f"{name}_n_simulations", n_sim):
            initial_state = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        prng_key = numpyro.prng_key()
        if prng_key is None:
            raise ValueError("PRNG key required for simulation")
        keys = jr.split(prng_key, n_sim)

        def _sim_one_trajectory(key, x0):
            keys_t = jr.split(key, T)

            def _step(x_prev, t_idx):
                t_now = times[t_idx]
                t_next = times[t_idx + 1]
                u_now = _get_val_or_None(ctrl_values, t_idx)
                u_next = _get_val_or_None(ctrl_values, t_idx + 1)
                trans_dist = state_transition(
                    x=x_prev,
                    u=u_now,
                    t_now=t_now,
                    t_next=t_next,
                )
                k_trans, k_obs = jr.split(keys_t[t_idx + 1], 2)
                x_t = trans_dist.sample(k_trans)
                y_t = dynamics.observation_model(x=x_t, u=u_next, t=t_next).sample(
                    k_obs
                )
                return x_t, (x_t, y_t)

            u_0 = _get_val_or_None(ctrl_values, 0)
            y_0 = dynamics.observation_model(x=x0, u=u_0, t=times[0]).sample(keys_t[0])
            _, (scan_states, scan_obs) = jax.lax.scan(_step, x0, jnp.arange(T - 1))
            states = jnp.concatenate([jnp.expand_dims(x0, 0), scan_states], axis=0)
            observations = jnp.concatenate([jnp.expand_dims(y_0, 0), scan_obs], axis=0)
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(
            keys, jnp.asarray(initial_state)
        )
        return {
            "times": _tile_times(times, n_sim),
            "states": _ensure_trailing_dim(states),
            "observations": _ensure_trailing_dim(observations),
        }


class ODESimulator(BaseSimulator):
    """Generation-only simulator for deterministic continuous-time dynamics."""

    def __init__(
        self,
        solver: dfx.AbstractSolver = dfx.Tsit5(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        dt0: float | int | Array = 1e-3,
        max_steps: int = 100_000,
        n_simulations: int = 1,
    ):
        """Configure ODE integration settings.

        Args:
            solver: Diffrax ODE solver (default: [`dfx.Tsit5`](https://docs.kidger.site/diffrax/api/solvers/ode_solvers/)).
                For solver guidance, see [How to choose a solver](https://docs.kidger.site/diffrax/usage/how-to-choose-a-solver/).
            adjoint: Diffrax adjoint strategy for differentiating through the ODE
                solve (relevant when used under gradient-based inference).
                See [Adjoints](https://docs.kidger.site/diffrax/api/adjoints/).
            stepsize_controller: Diffrax step-size controller (default:
                [`dfx.ConstantStepSize`](https://docs.kidger.site/diffrax/api/stepsize_controller/)).
            dt0: Initial step size (float or JAX array) passed to
                [`diffrax.diffeqsolve`](https://docs.kidger.site/diffrax/api/diffeqsolve/).
            max_steps: Hard cap on solver steps.
            n_simulations: Number of independent trajectory simulations. When > 1,
                states and observations have shape (n_simulations, T, ...).
        """
        dt0_arr = as_scalar_time_array(dt0, name="dt0")
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0_arr,
            "max_steps": max_steps,
        }
        self.n_simulations = n_simulations

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
        """Solve the ODE and sample observations in pure JAX after drawing `x0`."""
        del _obs_values_filled, _obs_mask, _obs_has_missing, kwargs
        times = obs_times if obs_times is not None else predict_times
        if times is None:
            raise ValueError("obs_times or predict_times must be provided")
        if obs_values is not None:
            raise ValueError(
                "ODESimulator is generation-only. Use predict_times for forward "
                "simulation, LatentStateBuilder for explicit latent-state "
                "inference, or Filter for marginalized inference."
            )

        n_sim = self.n_simulations

        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, times)
            control_path_eval = lambda t: control_path.evaluate(t, left=False)
        else:
            control_path_eval = lambda t: None

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]

        def _sim_one_trajectory(x0: Array, *, obs_key=None):
            """Simulate one ODE trajectory and sample its observations."""
            states = solve_ode(
                dynamics,
                t0,
                times,
                x0,
                control_path_eval,
                self.diffeqsolve_settings,
            )
            assert obs_key is not None
            observations = _emit_observations(
                dynamics,
                states,
                times,
                control_path_eval,
                obs_key,
            )
            return states, observations

        with numpyro.plate(f"{name}_n_simulations", n_sim):
            initial_state = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        prng_key = numpyro.prng_key()
        if prng_key is None:
            raise ValueError("PRNG key required for simulation")
        obs_keys = jr.split(prng_key, n_sim)
        states, observations = jax.vmap(_sim_one_trajectory)(
            jnp.asarray(initial_state), obs_key=obs_keys
        )
        return {
            "times": _tile_times(times, n_sim),
            "states": states,
            "observations": observations,
        }


def build_simulator_for_dynamics(
    dynamics: DynamicalModel,
    *,
    n_simulations: int = 1,
    simulator_config: SimulatorConfig | None = None,
) -> BaseSimulator:
    """Construct the concrete simulator implied by dynamics and config."""
    if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
        if isinstance(simulator_config, ODESimulatorConfig):
            raise ValueError(
                "ODESimulatorConfig is incompatible with stochastic continuous-time dynamics."
            )
        if isinstance(simulator_config, SDESimulatorConfig):
            return SDESimulator(
                solver=simulator_config.solver,
                stepsize_controller=simulator_config.stepsize_controller,
                adjoint=simulator_config.adjoint,
                dt0=simulator_config.dt0,
                tol_vbt=simulator_config.tol_vbt,
                max_steps=simulator_config.max_steps,
                n_simulations=n_simulations,
                source=simulator_config.source,
            )
        return SDESimulator(n_simulations=n_simulations)

    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if isinstance(simulator_config, SDESimulatorConfig):
            raise ValueError(
                "SDESimulatorConfig is incompatible with deterministic continuous-time dynamics."
            )
        if isinstance(simulator_config, ODESimulatorConfig):
            return ODESimulator(
                solver=simulator_config.solver,
                adjoint=simulator_config.adjoint,
                stepsize_controller=simulator_config.stepsize_controller,
                dt0=simulator_config.dt0,
                max_steps=simulator_config.max_steps,
                n_simulations=n_simulations,
            )
        return ODESimulator(n_simulations=n_simulations)

    if not dynamics.continuous_time:
        if isinstance(simulator_config, (ODESimulatorConfig, SDESimulatorConfig)):
            raise ValueError(
                "Continuous-time simulator configs are incompatible with discrete-time dynamics."
            )
        return DiscreteTimeSimulator(n_simulations=n_simulations)

    raise ValueError(
        f"Unsupported state evolution type: {type(dynamics.state_evolution)}. "
        "If using a generic function as a state evolution, you must specify the "
        "concrete simulator class manually."
    )


class Simulator(BaseSimulator):
    """Auto-selecting simulator wrapper.

    Chooses a concrete simulator based on the structure of `dynamics.state_evolution`:

    - `ContinuousTimeStateEvolution` with diffusion (and inferred `bm_dim`) -> `SDESimulator`
    - `ContinuousTimeStateEvolution` without diffusion -> `ODESimulator`
    - `DiscreteTimeStateEvolution` -> `DiscreteTimeSimulator`

    Note:
        - Auto-routing depends on structured model metadata (for example,
          `ContinuousTimeStateEvolution` vs. `DiscreteTimeStateEvolution`, and
          diffusion presence for continuous-time models).
        - If structure cannot be inferred (e.g., a generic callable state
          evolution), routing may fail and you should instantiate a concrete
          simulator class directly.

    Warning:
        The concrete simulator type is determined lazily on the **first call** and
        cached in ``self.simulator``. Re-using the same ``Simulator`` instance
        across models with different ``state_evolution`` types (e.g., first an ODE
        model, then an SDE model) will silently reuse the wrong backend. If you
        need to switch model types, create a new ``Simulator()`` instance.
    """

    def __init__(
        self,
        n_simulations: int = 1,
        simulator_config: SimulatorConfig | None = None,
    ):
        self.n_simulations = n_simulations
        self.simulator_config = simulator_config

        self.simulator: BaseSimulator | None = None

    def _simulate(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, Array]:
        if self.simulator is None:
            self.simulator = build_simulator_for_dynamics(
                dynamics,
                n_simulations=self.n_simulations,
                simulator_config=self.simulator_config,
            )

        return self.simulator._simulate(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            **kwargs,
        )
