"""NumPyro-aware simulators/unrollers for dynamical models."""

import dataclasses
import itertools
from collections.abc import Callable
from contextlib import contextmanager
from typing import cast

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpyro
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax import Array, lax
from numpyro.contrib.control_flow import scan as nscan

from dynestyx.handlers import HandlesSelf, _sample_intp
from dynestyx.inference.integrations.utils import WeightedParticles
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DiscreteTimeStateEvolution,
    DynamicalModel,
)
from dynestyx.types import FunctionOfTime, State
from dynestyx.utils import (
    _build_control_path,
    _get_val_or_None,
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


@contextmanager
def _suspend_numpyro_plate_frames():
    """Temporarily remove active numpyro.plate frames from the pyro stack."""
    stack = numpyro.primitives._PYRO_STACK
    original = list(stack)
    stack[:] = [f for f in original if not isinstance(f, numpyro.primitives.plate)]
    try:
        yield
    finally:
        stack[:] = original


def _array_has_plate_dims(
    arr: Array | None,
    plate_shapes: tuple[int, ...],
    *,
    min_suffix_ndim: int = 0,
) -> bool:
    """Return True if arr has leading dims exactly matching plate_shapes."""
    if arr is None:
        return False
    n_plates = len(plate_shapes)
    if arr.ndim < n_plates:
        return False
    for i, size in enumerate(plate_shapes):
        if arr.shape[i] != size:
            return False
    return (arr.ndim - n_plates) >= min_suffix_ndim


def _slice_array_for_plate_member(
    arr: Array | None, plate_shapes: tuple[int, ...], plate_idx: tuple[int, ...]
) -> Array | None:
    """Slice leading plate dims if present; otherwise return unchanged."""
    if arr is None:
        return None
    if _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1):
        return arr[plate_idx]
    return arr


def _slice_tree_for_plate_member(tree, plate_shapes: tuple[int, ...], plate_idx):
    """Slice all plate-batched array leaves in a pytree for one plate member."""

    def _is_distribution_leaf(node) -> bool:
        return isinstance(node, numpyro.distributions.Distribution)

    def _slice_leaf(leaf):
        if not isinstance(leaf, jax.Array):
            return leaf
        if not _array_has_plate_dims(leaf, plate_shapes, min_suffix_ndim=0):
            return leaf

        suffix_ndim = leaf.ndim - len(plate_shapes)
        # We slice two cases:
        # 1) suffix_ndim >= 2: classic batched tensors (e.g. A[M, d, d]).
        # 2) suffix_ndim == 0: per-member scalar params (e.g. beta[M]) inside
        #    nonlinear callable modules.
        #
        # We intentionally skip suffix_ndim == 1 to avoid ambiguous false
        # positives where unbatched vectors/matrices happen to start with a size
        # equal to the plate size (e.g. HMM state dim == plate size).
        if suffix_ndim == 0 or suffix_ndim >= 2:
            return leaf[plate_idx]
        return leaf

    return jax.tree.map(_slice_leaf, tree, is_leaf=_is_distribution_leaf)


def _dist_has_plate_batch_dims(dist_obj, plate_shapes: tuple[int, ...]) -> bool:
    """Return True when a distribution has plate-shaped leading batch dims."""
    if dist_obj is None or not hasattr(dist_obj, "batch_shape"):
        return False
    batch_shape = tuple(dist_obj.batch_shape)
    n_plates = len(plate_shapes)
    if len(batch_shape) < n_plates:
        return False
    for i, size in enumerate(plate_shapes):
        if batch_shape[i] != size:
            return False
    return True


def _slice_dist_for_plate_member(
    dist_obj, plate_shapes: tuple[int, ...], plate_idx: tuple[int, ...]
):
    """Slice plate-batched distribution parameters for one member."""
    if not _dist_has_plate_batch_dims(dist_obj, plate_shapes):
        return dist_obj

    def _slice_required_array(arr_like) -> Array:
        arr = jnp.asarray(arr_like)
        sliced = _slice_array_for_plate_member(arr, plate_shapes, plate_idx)
        if sliced is None:
            raise ValueError("Expected a concrete array when slicing plate member.")
        return sliced

    # Rebuild common distributions explicitly so cached/static batch metadata is
    # consistent after slicing.
    if isinstance(dist_obj, numpyro.distributions.MixtureSameFamily):
        mixture = _slice_dist_for_plate_member(
            dist_obj.mixing_distribution, plate_shapes, plate_idx
        )
        components = _slice_dist_for_plate_member(
            dist_obj.component_distribution, plate_shapes, plate_idx
        )
        return numpyro.distributions.MixtureSameFamily(mixture, components)

    if isinstance(dist_obj, numpyro.distributions.MultivariateNormal):
        loc = _slice_required_array(dist_obj.loc)
        cov = _slice_required_array(dist_obj.covariance_matrix)
        return numpyro.distributions.MultivariateNormal(
            loc=loc,
            covariance_matrix=cov,
        )

    if isinstance(dist_obj, numpyro.distributions.Delta):
        value = _slice_required_array(dist_obj.v)
        log_density = _slice_required_array(dist_obj.log_density)
        return numpyro.distributions.Delta(
            value,
            log_density=log_density,
            event_dim=dist_obj.event_dim,
        )

    if isinstance(dist_obj, WeightedParticles):
        particles = _slice_required_array(dist_obj.particles)
        log_weights = _slice_required_array(dist_obj.log_weights)
        return WeightedParticles(particles=particles, log_weights=log_weights)

    if dist_obj.__class__.__name__.startswith("Categorical"):
        if dist_obj.logits is not None:
            logits = _slice_required_array(dist_obj.logits)
            return numpyro.distributions.Categorical(logits=logits)
        probs = _slice_required_array(dist_obj.probs)
        return numpyro.distributions.Categorical(probs=probs)

    if isinstance(dist_obj, numpyro.distributions.Independent):
        base = _slice_dist_for_plate_member(dist_obj.base_dist, plate_shapes, plate_idx)
        return numpyro.distributions.Independent(
            base,
            dist_obj.reinterpreted_batch_ndims,
        )

    if isinstance(dist_obj, numpyro.distributions.TransformedDistribution):
        base = _slice_dist_for_plate_member(dist_obj.base_dist, plate_shapes, plate_idx)
        transforms = getattr(dist_obj, "transforms")
        return numpyro.distributions.TransformedDistribution(
            base,
            transforms,
        )

    def _slice_leaf(leaf):
        if isinstance(leaf, jax.Array) and _array_has_plate_dims(
            leaf, plate_shapes, min_suffix_ndim=1
        ):
            return leaf[plate_idx]
        return leaf

    return jax.tree.map(_slice_leaf, dist_obj)


def _stack_member_results(
    member_results: list[dict[str, Array]], plate_shapes: tuple[int, ...]
) -> dict[str, Array]:
    """Stack per-member simulator outputs into leading plate dimensions."""
    if not member_results:
        return {}

    keys = member_results[0].keys()
    for result in member_results:
        if result.keys() != keys:
            raise ValueError(
                "Plate simulator members returned inconsistent result keys."
            )

    stacked = {}
    for key in keys:
        values = [r[key] for r in member_results]
        flat = jnp.stack(values, axis=0)
        stacked[key] = flat.reshape(*plate_shapes, *values[0].shape)
    return stacked


def _member_name(base_name: str, plate_idx: tuple[int, ...]) -> str:
    """Stable member prefix used for plate-mode simulator sample sites."""
    joined = "_".join(str(i) for i in plate_idx)
    return f"{base_name}_p{joined}"


def _tree_has_plate_batched_leaf(tree, plate_shapes: tuple[int, ...]) -> bool:
    """Return True if any JAX-array leaf has leading plate_shapes."""

    def _is_distribution_leaf(node) -> bool:
        return isinstance(node, numpyro.distributions.Distribution)

    for leaf in jax.tree.leaves(tree, is_leaf=_is_distribution_leaf):
        if not isinstance(leaf, jax.Array):
            continue
        if not _array_has_plate_dims(leaf, plate_shapes, min_suffix_ndim=0):
            continue
        suffix_ndim = leaf.ndim - len(plate_shapes)
        if suffix_ndim == 0 or suffix_ndim >= 2:
            return True
    return False


def _ensure_continuous_bm_dim(dynamics: DynamicalModel) -> DynamicalModel:
    """Infer and set bm_dim when continuous dynamics were constructed inside a plate."""
    if not dynamics.continuous_time:
        return dynamics

    state_evolution = dynamics.state_evolution
    if (
        not isinstance(state_evolution, ContinuousTimeStateEvolution)
        or state_evolution.diffusion_coefficient is None
        or state_evolution.bm_dim is not None
    ):
        return dynamics

    x0 = jnp.zeros((dynamics.state_dim,))
    u0 = None if dynamics.control_dim == 0 else jnp.zeros((dynamics.control_dim,))
    t0 = jnp.array(0.0) if dynamics.t0 is None else jnp.asarray(dynamics.t0)
    diffusion_shape = jax.eval_shape(
        lambda: state_evolution.diffusion_coefficient(x0, u0, t0)
    ).shape
    if len(diffusion_shape) != 2:
        raise ValueError(
            "diffusion_coefficient must return shape (state_dim, bm_dim). "
            f"Got shape {diffusion_shape}."
        )
    if int(diffusion_shape[0]) != int(dynamics.state_dim):
        raise ValueError(
            "diffusion_coefficient first dimension must match state_dim. "
            f"Got diffusion_shape={diffusion_shape}, state_dim={dynamics.state_dim}."
        )
    inferred_bm_dim = int(diffusion_shape[1])
    object.__setattr__(state_evolution, "bm_dim", inferred_bm_dim)
    return dynamics


class BaseSimulator(ObjectInterpretation, HandlesSelf):
    """Base class for simulator/unroller handlers.

    Interprets `dsx.sample(name, dynamics, obs_times=..., obs_values=..., ...)` by
    unrolling `dynamics` into NumPyro sample sites (latent states and emissions) on
    the provided time grid.

    When the simulator runs, it records the solved trajectories as deterministic
    sites (conventionally `"times"`, `"states"`, and `"observations"`).

    Notes:
        - If `obs_times` is None, the handler is a no-op.
        - If `obs_values` is provided, observation sample sites are conditioned via
          `obs=...`.
        - Conditioning (`obs_values is not None`) is only supported for
          ``n_simulations == 1``.  Subclasses that permit conditioning enforce this
          via the base-class guard in ``_sample_ds``; they do not need to duplicate
          the check themselves.
    """

    n_simulations: int = 1

    def _run_single_member_simulation(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        filtered_times=None,
        filtered_dists=None,
        **kwargs,
    ) -> dict[str, Array] | None:
        """Run simulator logic for one unbatched member and return trajectories."""
        dynamics = _ensure_continuous_bm_dim(dynamics)

        if (
            filtered_times is not None
            and filtered_dists is None
            and predict_times is not None
        ):
            raise ValueError(
                "Rollout requested with filtered_times but missing filtered_dists. "
                "Plate-aware rollout requires filtered distributions from the filter."
            )

        # Need times to simulate: predict_times or obs_times
        # For filter rollout, need predict_times
        if predict_times is None:
            if obs_times is None or filtered_times is not None:
                return None

        filter_rollout = filtered_times is not None and filtered_dists is not None

        if filter_rollout:
            _validate_site_sorting(filtered_times, name="filtered_times")
            n_pred = len(predict_times)

            # Build segment ids on host once.
            # seg_id == -1 means "before first filtered time" (use model prior).
            pt_host = np.asarray(jax.device_get(predict_times))
            ft_host = np.asarray(jax.device_get(filtered_times))
            seg_ids_host = np.searchsorted(ft_host, pt_host, side="right") - 1

            def _ctrl_for_segment(sub_times):
                if ctrl_times is None or ctrl_values is None:
                    return None, None
                inds = jnp.searchsorted(ctrl_times, sub_times, side="left")
                return sub_times, ctrl_values[inds]

            def _dynamics_for_segment(seg_id: int):
                if seg_id < 0:
                    return dynamics, f"{name}_0"

                filtered_time = filtered_times[seg_id]
                filtered_dist = filtered_dists[seg_id]
                dynamics_with_filtered_time = eqx.tree_at(
                    lambda m: m.t0,
                    dynamics,
                    filtered_time,
                    is_leaf=lambda x: x is None,
                )
                dynamics_seg = eqx.tree_at(
                    lambda m: m.initial_condition,
                    dynamics_with_filtered_time,
                    filtered_dist,
                    is_leaf=lambda x: x is None,
                )
                return dynamics_seg, f"{name}_{seg_id + 1}"

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
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        filtered_times=None,
        filtered_dists=None,
        **kwargs,
    ) -> dict[str, Array] | None:
        """Run simulator over all plate members and stack outputs."""
        has_batched_source = _tree_has_plate_batched_leaf(
            dynamics, plate_shapes
        ) or any(
            _array_has_plate_dims(arr, plate_shapes, min_suffix_ndim=1)
            for arr in (
                obs_times,
                obs_values,
                ctrl_times,
                ctrl_values,
                predict_times,
                filtered_times,
            )
        )
        if (not has_batched_source) and filtered_dists is not None:
            has_batched_source = any(
                _dist_has_plate_batch_dims(dist_obj, plate_shapes)
                for dist_obj in filtered_dists
            )
        if not has_batched_source:
            raise ValueError(
                "Plate simulator received plate_shapes but no plate-batched dynamics/data "
                "sources were found. At least one source must have leading dimensions "
                "matching plate_shapes."
            )

        plate_indices = list(itertools.product(*[range(s) for s in plate_shapes]))
        member_results: list[dict[str, Array]] = []

        for plate_idx in plate_indices:
            member_name = _member_name(name, plate_idx)
            member_dynamics = _slice_tree_for_plate_member(
                dynamics, plate_shapes, plate_idx
            )
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
            member_obs_times = _slice_array_for_plate_member(
                obs_times, plate_shapes, plate_idx
            )
            member_obs_values = _slice_array_for_plate_member(
                obs_values, plate_shapes, plate_idx
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
            member_filtered_dists = None
            if filtered_dists is not None:
                member_filtered_dists = [
                    _slice_dist_for_plate_member(d, plate_shapes, plate_idx)
                    for d in filtered_dists
                ]

            with _suspend_numpyro_plate_frames():
                member_result = self._run_single_member_simulation(
                    member_name,
                    member_dynamics,
                    obs_times=member_obs_times,
                    obs_values=member_obs_values,
                    ctrl_times=member_ctrl_times,
                    ctrl_values=member_ctrl_values,
                    predict_times=member_predict_times,
                    filtered_times=member_filtered_times,
                    filtered_dists=member_filtered_dists,
                    **kwargs,
                )

            if member_result is not None:
                member_results.append(member_result)

        if not member_results:
            return None
        return _stack_member_results(member_results, plate_shapes)

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        plate_shapes=(),
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        filtered_times=None,
        filtered_dists=None,
        **kwargs,
    ) -> FunctionOfTime:
        if plate_shapes:
            results = self._run_plated_simulation(
                name,
                dynamics,
                plate_shapes=plate_shapes,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=predict_times,
                filtered_times=filtered_times,
                filtered_dists=filtered_dists,
                **kwargs,
            )
        else:
            results = self._run_single_member_simulation(
                name,
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=predict_times,
                filtered_times=filtered_times,
                filtered_dists=filtered_dists,
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
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, State]:
        """Unroll `dynamics` as a NumPyro model.

        Implementations are expected to:
        - require `obs_times` (the grid at which to simulate and emit observations),
        - sample (and possibly condition) observation sites using `obs_values`,
        - and return arrays suitable for recording as deterministic sites.

        Args:
            dynamics: Dynamical model to simulate/unroll.
            obs_times: Observation times. Required by all concrete simulators.
            obs_values: Optional observations. If provided, observation sites
                are conditioned via `obs=...`.
            ctrl_times: Optional control times.
            ctrl_values: Optional control values aligned to `ctrl_times`.
            predict_times: Optional prediction times. If provided, prediction sites are
                emitted at those times as `numpyro.sample("y_i", ..., obs=None)`.
        Returns:
            dict[str, State]: Mapping from deterministic site names to
                trajectories. Conventionally includes `"times"`, `"states"`,
                and `"observations"`.
        """
        raise NotImplementedError()


def _solve_de(
    dynamics,
    t0: float,
    saveat_times: Array,
    x0: State,
    control_path_eval: Callable[[Array], Array | None],
    diffeqsolve_settings: dict,
    *,
    key=None,
    tol_vbt: float | None = None,
) -> Array:
    """Solve one ODE/SDE trajectory with diffrax.

    Uses ODE mode when diffusion is None, otherwise SDE mode. `t0` is explicit
    so rollout segments can start from filtered times.
    """
    t1 = saveat_times[-1]

    # Keep the branch JAX-traceable when t0/t1 are traced.
    def _early_return():
        return jnp.broadcast_to(x0, (len(saveat_times),) + jnp.shape(x0))

    def _solve():
        diffusion = dynamics.state_evolution.diffusion_coefficient

        def _drift(t, y, args):
            u_t = args(t) if args is not None else None
            return dynamics.state_evolution.total_drift(x=y, u=u_t, t=t)

        if diffusion is None:
            terms = dfx.ODETerm(_drift)
        else:
            k_bm, _ = jr.split(key, 2)
            bm = dfx.VirtualBrownianTree(
                t0=t0,
                t1=t1,
                tol=tol_vbt,
                shape=(dynamics.state_evolution.bm_dim,),
                key=k_bm,
            )

            def _diffusion(t, y, args):
                u_t = args(t) if args is not None else None
                return dynamics.state_evolution.diffusion_coefficient(x=y, u=u_t, t=t)

            terms = dfx.MultiTerm(  # type: ignore
                dfx.ODETerm(_drift), dfx.ControlTerm(_diffusion, bm)
            )

        sol = dfx.diffeqsolve(
            terms,
            t0=t0,
            t1=t1,
            y0=x0,
            saveat=dfx.SaveAt(ts=saveat_times),
            args=control_path_eval,
            **diffeqsolve_settings,
        )
        return sol.ys

    return lax.cond(t0 >= t1, _early_return, _solve)


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


class SDESimulator(BaseSimulator):
    """Simulator for continuous-time stochastic dynamics (SDEs).

    This simulator integrates a `ContinuousTimeStateEvolution` with nonzero diffusion
    using Diffrax and a `VirtualBrownianTree` (see the Diffrax docs on
    [Brownian controls](https://docs.kidger.site/diffrax/api/brownian/)). It constructs a NumPyro generative
    model with state sample sites (starting at `"x_0"`) and observation sample sites
    (`"y_0"`, `"y_1"`, ...).

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
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver = dfx.Heun(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        dt0: float = 1e-4,
        tol_vbt: float | None = None,
        max_steps: int | None = None,
        n_simulations: int = 1,
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
            dt0: Initial step size passed to
                [`diffrax.diffeqsolve`](https://docs.kidger.site/diffrax/api/diffeqsolve/).
            tol_vbt: Tolerance parameter for
                [`diffrax.VirtualBrownianTree`](https://docs.kidger.site/diffrax/api/brownian/). If None,
                defaults to `dt0 / 2`. For statistically correct simulation, this
                must be smaller than `dt0`.
            max_steps: Optional hard cap on solver steps.
            n_simulations: Number of independent trajectory simulations. When > 1,
                states and observations have an extra leading dimension (n_simulations, T, ...).

        Notes:
            - `VirtualBrownianTree` draws randomness via `numpyro.prng_key()`, so
              `SDESimulator` must be executed inside a seeded NumPyro context.
        """
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
            "max_steps": max_steps,
        }
        self.n_simulations = n_simulations

        if tol_vbt is None:
            self.tol_vbt = dt0 / 2.0
        else:
            self.tol_vbt = tol_vbt

        assert self.tol_vbt < dt0, (
            "tol_vbt must be smaller than dt0 for statistically correct simulation."
        )

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
    ) -> dict[str, State]:
        """
        Unroll a continuous-time SDE as a NumPyro model.

        This method:
        - samples the initial latent state as `numpyro.sample("x_0", ...)`,
        - integrates the SDE to all `obs_times` using Diffrax,
        - emits observations at those times as `numpyro.sample("y_i", ..., obs=...)`,
        - and returns trajectories for deterministic recording.

        To handle controls, we use a rectilinear interpolation that is right-continuous,
        i.e., if ctrl_times = [0.0, 1.0, 2.0] and ctrl_values = [0.0, 1.0, 2.0],
        then the control at time 1.0 is the value at time 1.0.

        Args:
            dynamics: A `DynamicalModel` whose `state_evolution` is a
                `ContinuousTimeStateEvolution` with a non-None diffusion coefficient
                and inferred `bm_dim` (set during `DynamicalModel` construction).
            obs_times: Times at which to save the latent state and emit observations.
                Required.
            obs_values: Optional observation array. If provided, observation sites are
                conditioned via `obs=obs_values[i]`.
            ctrl_times: Optional control times.
            ctrl_values: Optional control values aligned to `ctrl_times`.
            predict_times: Optional prediction times. If provided, prediction sites are
                emitted at those times as `numpyro.sample("y_i", ..., obs=None)`.
        Returns:
            dict[str, State]: Dictionary with `"times"`, `"states"`, and
                `"observations"` trajectories.

        Warning:
            Conditioning on `obs_values` here is generally **not** a good way to do
            parameter inference for SDEs, because it introduces an explicit, high-
            dimensional latent path. Prefer filtering (`Filter`) or particle methods.
        """
        if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            raise NotImplementedError(
                f"SDESimulator only works with ContinuousTimeStateEvolution, got {type(dynamics.state_evolution)}"
            )

        if dynamics.state_evolution.diffusion_coefficient is None:
            raise ValueError(
                "SDESimulator requires diffusion_coefficient to be defined "
                f"(got coeff={dynamics.state_evolution.diffusion_coefficient}). "
                "Use ODESimulator for deterministic dynamics."
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
            """Simulate one SDE trajectory and its emissions."""
            k_solve, k_obs = jr.split(key, 2)
            states_sol = _solve_de(
                dynamics,
                t0,
                times,
                x0,
                control_path_eval,
                self.diffeqsolve_settings,
                key=k_solve,
                tol_vbt=self.tol_vbt,
            )
            emissions = _emit_observations(
                name, dynamics, states_sol, times, None, control_path_eval, key=k_obs
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
    """Simulator for discrete-time dynamical models.

    n_simulations: Number of independent trajectory simulations. When > 1,
        states and observations have an extra leading dimension (n_simulations, T, ...).
        Only supported when obs_values is None (forward simulation).

    This unrolls a discrete-time `DynamicalModel` as a NumPyro model:

    - samples an initial state (`"x_0"`),
    - repeatedly samples transitions (`"x_1"`, `"x_2"`, ...) and observations
      (`"y_0"`, `"y_1"`, ...),
    - and, if provided, conditions on `obs_values` via `obs=...`.

    Optimization for fully observed state:
        If `dynamics.observation_model` is `DiracIdentityObservation` and
        `obs_values` is provided, then $y_t = x_t$ and the latent state is
        observed directly. In this case, the simulator:

        - conditions the initial state as `numpyro.sample("x_0", ..., obs=obs_values[0])`,
        - records `"y_0"` deterministically,
        - and vectorizes the transition likelihood across time using a
          `numpyro.plate("time", T-1)` rather than a scan, for efficiency.

        The returned `"states"` and `"observations"` are both `obs_values`.

    Deterministic outputs:
        When run, the simulator records `"times"`, `"states"`, and `"observations"`
        as `numpyro.deterministic(...)` sites.

    """

    n_simulations: int = 1

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
    ) -> dict[str, State]:
        """Unroll a discrete-time model as a NumPyro model.

        Creates NumPyro sample sites for the initial condition (`"x_0"`), subsequent
        states (`"x_1"`, ...), and observations (`"y_0"`, ...). If `obs_values` is
        provided, observation sites are conditioned via `obs=...`.

        Notes:
            - For `DiracIdentityObservation` with provided `obs_values`, the latent
              state is observed directly (`y_t = x_t`) and this uses a plated
              transition likelihood instead of a scan for efficiency.

        Args:
            dynamics: Discrete-time `DynamicalModel` to unroll.
            obs_times: Discrete observation indices/times. Required.
            obs_values: Optional observations for conditioning.
            ctrl_times: Optional control times.
            ctrl_values: Optional controls aligned to `ctrl_times`.
            predict_times: Optional prediction times. If provided, prediction sites are
                emitted at those times as `numpyro.sample("y_i", ..., obs=None)`.
        Returns:
            dict[str, State]: Dictionary with `"times"`, `"states"`, and
                `"observations"` trajectories.
        """
        times = obs_times if obs_times is not None else predict_times
        if times is None:
            raise ValueError("obs_times or predict_times must be provided")

        T = len(times)
        if T < 1:
            raise ValueError("obs_times must contain at least one timepoint")

        n_sim = self.n_simulations

        # DiracIdentityObservation with observed values: y_t = x_t, so we use plating
        # instead of scan. state_evolution returns a dist; call it with batched inputs.
        if isinstance(dynamics.observation_model, DiracIdentityObservation) and (
            obs_values is not None
        ):
            with numpyro.plate(f"{name}_n_simulations", 1):
                numpyro.sample(
                    f"{name}_x_0",
                    dynamics.initial_condition,
                    obs=jnp.expand_dims(obs_values[0], axis=0),
                )
            numpyro.deterministic(f"{name}_y_0", jnp.expand_dims(obs_values[0], axis=0))
            if T == 1:
                # No transitions exist for a single-timepoint trajectory.
                # Always return (n_sim, T, state_dim) for consistent shaping
                obs_exp = _ensure_trailing_dim(jnp.expand_dims(obs_values, axis=0))
                return {
                    "times": _tile_times(times, 1),
                    "states": obs_exp,
                    "observations": obs_exp,
                }

            # Ensure (T-1, state_dim) so swapaxes to (state_dim, T-1) is valid (state_dim=1 => 1D otherwise).
            if obs_values.ndim == 1:
                x_prev = obs_values[:-1][:, None]
                x_next = obs_values[1:][:, None]
            else:
                x_prev = obs_values[:-1]
                x_next = obs_values[1:]
            if ctrl_values is not None:
                if ctrl_values.ndim == 1:
                    u_prev = ctrl_values[:-1][:, None]
                else:
                    u_prev = ctrl_values[:-1]
            else:
                u_prev = None
            t_now = times[:-1]
            t_next = times[1:]

            # Pass state (and controls) with batch as last axis so drift can use
            # naive indexing (x[0], x[1], ...) and discretizer broadcasts correctly.
            x_prev_batch_last = jnp.swapaxes(x_prev, 0, 1)
            x_next_batch_last = jnp.swapaxes(x_next, 0, 1)
            u_prev_batch_last = (
                jnp.swapaxes(u_prev, 0, 1) if u_prev is not None else None
            )

            with numpyro.plate("time", T - 1):
                trans = dynamics.state_evolution(
                    x_prev_batch_last,
                    u_prev_batch_last,
                    t_now,
                    t_next,  # type: ignore
                )
                # obs shape must match trans.batch_shape + trans.event_shape: use
                # time-first (T-1, state_dim) for e.g. discretizer; batch-last (state_dim, T-1) for scalar.
                obs_next = x_next_batch_last if dynamics.state_dim == 1 else x_next
                numpyro.sample("x_next", trans, obs=obs_next)  # type: ignore

            # Always return (n_sim, T, state_dim) for consistent shaping
            obs_exp = _ensure_trailing_dim(jnp.expand_dims(obs_values, axis=0))
            return {
                "times": _tile_times(times, 1),
                "states": obs_exp,
                "observations": obs_exp,
            }

        def _step_dists(x_prev, t_idx):
            """Compute transition distribution and step metadata from a single state."""
            t_now = times[t_idx]
            t_next = times[t_idx + 1]
            u_now = _get_val_or_None(ctrl_values, t_idx)
            u_next = _get_val_or_None(ctrl_values, t_idx + 1)
            trans_dist = dynamics.state_evolution(
                x=x_prev, u=u_now, t_now=t_now, t_next=t_next
            )
            return t_next, u_next, trans_dist

        # n_simulations > 1: vmapped pure-JAX loop; avoid numpyro.sample in vmap body.
        # can't do obs= conditioning with the lax.scan.
        if n_sim > 1:
            with numpyro.plate(f"{name}_n_simulations", n_sim):
                initial_state = numpyro.sample(
                    f"{name}_x_0", dynamics.initial_condition
                )
            prng_key = numpyro.prng_key()
            if prng_key is None:
                raise ValueError("PRNG key required for n_simulations > 1")
            keys = jr.split(prng_key, n_sim)

            def _sim_one_trajectory(key, x0):
                """Simulate one discrete trajectory and its emissions."""
                keys_t = jr.split(key, T)

                def _step(carry, t_idx):
                    x_prev = carry
                    k_trans, k_obs = jr.split(keys_t[t_idx], 2)
                    t_next, u_next, trans_dist = _step_dists(x_prev, t_idx)
                    x_t = trans_dist.sample(k_trans)
                    y_t = dynamics.observation_model(x=x_t, u=u_next, t=t_next).sample(
                        k_obs
                    )
                    return x_t, (x_t, y_t)

                u_0 = _get_val_or_None(ctrl_values, 0)
                y_0 = dynamics.observation_model(x=x0, u=u_0, t=times[0]).sample(
                    keys_t[0]
                )
                _, (scan_states, scan_obs) = jax.lax.scan(_step, x0, jnp.arange(T - 1))
                states = jnp.concatenate([jnp.expand_dims(x0, 0), scan_states], axis=0)
                observations = jnp.concatenate(
                    [jnp.expand_dims(y_0, 0), scan_obs], axis=0
                )
                return states, observations

            states, observations = jax.vmap(_sim_one_trajectory)(keys, initial_state)
            return {
                "times": _tile_times(times, n_sim),
                "states": _ensure_trailing_dim(states),
                "observations": _ensure_trailing_dim(observations),
            }

        # Default: scan over time (n_simulations == 1)...allows for obs= conditioning.
        with numpyro.plate(f"{name}_n_simulations", 1):
            x_prev_site: State = numpyro.sample(  # type: ignore
                f"{name}_x_0", dynamics.initial_condition
            )
        x_prev = x_prev_site[0]

        u_0 = _get_val_or_None(ctrl_values, 0)
        obs_0 = _get_val_or_None(obs_values, 0)
        if obs_0 is not None:
            obs_0 = jnp.expand_dims(obs_0, axis=0)
        with numpyro.plate(f"{name}_n_simulations", 1):
            y_0_site = numpyro.sample(
                f"{name}_y_0",
                dynamics.observation_model(x_prev, u_0, times[0]),
                obs=obs_0,
            )
        y_0_arr = cast(Array, jnp.asarray(y_0_site))
        y_0 = y_0_arr[0]

        def _step(x_prev, t_idx):
            t_next, u_next, trans_dist = _step_dists(x_prev, t_idx)
            with numpyro.plate(f"{name}_n_simulations", 1):
                x_t_site = numpyro.sample(f"{name}_x_{t_idx + 1}", trans_dist)
            x_t = x_t_site[0]
            obs_next = _get_val_or_None(obs_values, t_idx + 1)
            if obs_next is not None:
                obs_next = jnp.expand_dims(obs_next, axis=0)
            with numpyro.plate(f"{name}_n_simulations", 1):
                y_t_site = numpyro.sample(
                    f"{name}_y_{t_idx + 1}",
                    dynamics.observation_model(x=x_t, u=u_next, t=t_next),
                    obs=obs_next,
                )
            y_t = y_t_site[0]
            return x_t, (x_t, y_t)

        _, scan_outputs = nscan(_step, x_prev, jnp.arange(T - 1))
        scan_states, scan_observations = scan_outputs

        x_0_expanded = jnp.expand_dims(x_prev, axis=0)  # type: ignore
        y_0_expanded = jnp.expand_dims(y_0, axis=0)
        states = jnp.concatenate([x_0_expanded, scan_states], axis=0)
        observations = jnp.concatenate([y_0_expanded, scan_observations], axis=0)
        # Always return (n_sim, T, state_dim) for consistent shaping
        return {
            "times": _tile_times(times, 1),
            "states": _ensure_trailing_dim(jnp.expand_dims(states, axis=0)),
            "observations": _ensure_trailing_dim(jnp.expand_dims(observations, axis=0)),
        }


class ODESimulator(BaseSimulator):
    """Simulator for continuous-time deterministic dynamics (ODEs).

    This unrolls a `ContinuousTimeStateEvolution` with **no diffusion** by solving
    an ODE using Diffrax and then emitting observations at `obs_times` as NumPyro
    sample sites. Solver options can be configured via the constructor.

    n_simulations: Number of independent trajectory simulations. When > 1,
        samples multiple initial conditions and runs the ODE from each; states
        and observations have shape (n_simulations, T, ...). When 1, shape is
        (1, T, ...) for consistency.

    Controls:
        If `ctrl_times` / `ctrl_values` are provided at the `dsx.sample(...)` site,
        controls are interpolated with a right-continuous rectilinear rule
        (`left=False`), i.e., the control at time `t_k` is `ctrl_values[k]`.

    Conditioning:
        If `obs_values` is provided, observation sites are conditioned via `obs=...`.

    Deterministic outputs:
        When run, the simulator records `"times"`, `"states"`, and `"observations"`
        as `numpyro.deterministic(...)` sites.
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver = dfx.Tsit5(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        dt0: float = 1e-3,
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
            dt0: Initial step size passed to
                [`diffrax.diffeqsolve`](https://docs.kidger.site/diffrax/api/diffeqsolve/).
            max_steps: Hard cap on solver steps.
            n_simulations: Number of independent trajectory simulations. When > 1,
                states and observations have shape (n_simulations, T, ...).
        """
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
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
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> dict[str, State]:
        """Unroll a deterministic continuous-time model as a NumPyro model.

        This method:
        - samples the initial state as `numpyro.sample("x_0", ...)`,
        - solves the ODE and saves the solution at the time grid,
        - emits observations as `numpyro.sample("y_i", ..., obs=...)`.

        Args:
            dynamics: A `DynamicalModel` whose `state_evolution` is a
                `ContinuousTimeStateEvolution` with deterministic dynamics.
            obs_times: Times at which to save the latent state and emit observations.
            obs_values: Optional observation array. If provided, observation sites are
                conditioned via `obs=obs_values[i]`.
            ctrl_times: Optional control times.
            ctrl_values: Optional controls aligned to `ctrl_times`.
            predict_times: Used when obs_times is None (e.g. from Filter).

        Returns:
            dict[str, State]: Dictionary with `"times"`, `"states"`, and
                `"observations"` trajectories.
        """
        times = obs_times if obs_times is not None else predict_times
        if times is None:
            raise ValueError("obs_times or predict_times must be provided")

        n_sim = self.n_simulations

        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, times)
            control_path_eval = lambda t: control_path.evaluate(t, left=False)
        else:
            control_path_eval = lambda t: None

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]

        def _sim_one_trajectory(x0: Array, *, obs_key=None):
            """Simulate one ODE trajectory and emit observations."""
            states = _solve_de(
                dynamics,
                t0,
                times,
                x0,
                control_path_eval,
                self.diffeqsolve_settings,
            )
            observations = _emit_observations(
                name,
                dynamics,
                states,
                times,
                obs_values,
                control_path_eval,
                key=obs_key,
            )
            return states, observations

        if obs_values is not None:
            # Conditioning mode (n_sim must be 1 due to guard above).
            # Uses numpyro.sample per observation site to support obs= conditioning.
            with numpyro.plate(f"{name}_n_simulations", 1):
                x0 = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
            x0_arr: Array = jnp.asarray(x0)[0]
            states, observations = _sim_one_trajectory(x0_arr)
            return {
                "times": _tile_times(times, 1),
                "states": jnp.expand_dims(states, axis=0),
                "observations": jnp.expand_dims(observations, axis=0),
            }

        # Forward simulation (obs_values is None): vmap over all n_sim, including 1.
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


class Simulator(BaseSimulator):
    """Auto-selecting simulator wrapper.

    Chooses a concrete simulator based on the structure of `dynamics.state_evolution`:

    - `ContinuousTimeStateEvolution` with diffusion (and inferred `bm_dim`) -> `SDESimulator`
    - `ContinuousTimeStateEvolution` without diffusion -> `ODESimulator`
    - `DiscreteTimeStateEvolution` -> `DiscreteTimeSimulator`

    Note:
        - Any `*args` / `**kwargs` are forwarded to the routed simulator
          constructor, so Diffrax settings can be supplied here when routing to
          `ODESimulator` / `SDESimulator`.
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

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

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
    ) -> dict[str, State]:
        if self.simulator is None:
            if isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
                if dynamics.state_evolution.diffusion_coefficient is None:
                    self.simulator = ODESimulator(*self.args, **self.kwargs)
                else:
                    self.simulator = SDESimulator(*self.args, **self.kwargs)
            elif isinstance(dynamics.state_evolution, DiscreteTimeStateEvolution):
                self.simulator = DiscreteTimeSimulator(*self.args, **self.kwargs)
            else:
                raise ValueError(
                    f"Unsupported state evolution type: {type(dynamics.state_evolution)}."
                    + "If using a generic function as a state evolution, you must specify the type of simulator manually."
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
