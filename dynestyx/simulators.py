"""NumPyro-aware simulators/unrollers for dynamical models."""

import dataclasses
import warnings
from collections.abc import Callable
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
    """Ensure simulator output has a trailing state/obs dimension.

    All simulator outputs should have shape ``(n_sim, T, dim)``.  For scalar
    states or observations (e.g. HMM discrete latent variables), the array
    arrives with shape ``(n_sim, T)`` — ndim==2.  This adds a trailing
    singleton to give ``(n_sim, T, 1)``, consistent with the ndim==3 contract
    required by ``_merge_segments`` and the rest of the pipeline.  A future
    one-hot encoding upgrade can then widen that last axis to ``num_states``.

    Arrays that already carry a trailing dimension (ndim==3) are returned
    unchanged.
    """
    return arr[..., jnp.newaxis] if arr.ndim == 2 else arr


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
    """

    @implements(_sample_intp)
    def _sample_ds(
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
    ) -> FunctionOfTime:
        # Need times to simulate: predict_times or obs_times
        # For filter rollout, need predict_times
        if predict_times is None:
            if obs_times is None or filtered_times is not None:
                return fwd(name, dynamics, **kwargs)

        if filtered_times is not None and filtered_dists is not None:
            _validate_site_sorting(filtered_times, name="filtered_times")
            n_pred = len(predict_times)

            def _extract_times(mask: Array, empty_fill: Array) -> Array:
                """JAX-safe fixed-size extract used in tracer fallback path.

                This path preserves static shapes under lax.map/jit, but can be slower
                because empty segments still evaluate with full-size `predict_times`.
                """
                n_valid = jnp.sum(mask)
                valid_max = jnp.max(jnp.where(mask, predict_times, -jnp.inf))
                fill = jnp.where(jnp.isfinite(valid_max), valid_max, empty_fill)
                extracted = jnp.extract(
                    mask, predict_times, size=n_pred, fill_value=fill
                )
                # Pad positions must be strictly increasing for ODE/SDE control paths
                pad_start = fill + 1
                is_valid_pos = jnp.arange(n_pred) < n_valid
                pad_vals = pad_start + (jnp.arange(n_pred) - n_valid)
                result = jnp.where(is_valid_pos, extracted, pad_vals)
                return lax.cond(n_valid > 0, lambda: result, lambda: predict_times)

            def _ctrl_for_segment(sub_times):
                if ctrl_times is None or ctrl_values is None:
                    return None, None
                inds = jnp.searchsorted(ctrl_times, sub_times, side="left")
                return sub_times, ctrl_values[inds]

            sim_results = []
            seg_masks = []

            # Fast path: use host-side segment IDs so we only simulate non-empty segments
            # and pass true segment lengths (sum lengths = n_pred).
            # This avoids O(n_filtered * n_pred) behavior.
            used_fast_path = False
            try:
                pt_host = np.asarray(jax.device_get(predict_times))
                ft_host = np.asarray(jax.device_get(filtered_times))
                seg_ids_host = np.searchsorted(ft_host, pt_host, side="right") - 1
                present_seg_ids = [int(s) for s in np.unique(seg_ids_host)]

                for seg_id in present_seg_ids:
                    mask_host = seg_ids_host == seg_id
                    if not np.any(mask_host):
                        continue

                    mask_seg = jnp.asarray(mask_host)
                    sub = jnp.asarray(pt_host[mask_host], dtype=predict_times.dtype)

                    if seg_id < 0:
                        dynamics_seg = dynamics
                        seg_name = f"{name}_0"
                    else:
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
                        seg_name = f"{name}_{seg_id + 1}"

                    ctrl_t_seg, ctrl_v_seg = _ctrl_for_segment(sub)
                    sim_results.append(
                        self._simulate(
                            seg_name,
                            dynamics_seg,
                            obs_times=None,
                            obs_values=None,
                            ctrl_times=ctrl_t_seg,
                            ctrl_values=ctrl_v_seg,
                            predict_times=sub,
                        )
                    )
                    seg_masks.append(mask_seg)

                used_fast_path = len(sim_results) > 0
            except Exception:
                used_fast_path = False

            if not used_fast_path:
                # Tracer-safe fallback path with fixed-size segments.
                # First segment: times before first filtered time
                mask0 = predict_times < filtered_times[0]
                seg_masks.append(mask0)  # for filtering concatenated output
                sub0 = _extract_times(mask0, empty_fill=filtered_times[0])
                ctrl_t_seg, ctrl_v_seg = _ctrl_for_segment(sub0)
                sim_results.append(
                    self._simulate(
                        f"{name}_0",
                        dynamics,
                        obs_times=None,
                        obs_values=None,
                        ctrl_times=ctrl_t_seg,
                        ctrl_values=ctrl_v_seg,
                        predict_times=sub0,
                    )
                )

                # Segments between filtered times (index-based to avoid JAX array chunk iterator
                # which fails under lax.map when len(filtered_times) > 100)
                n_filtered = int(filtered_times.shape[0])
                for f_idx in range(n_filtered):
                    filtered_time = filtered_times[f_idx]
                    filtered_dist = filtered_dists[f_idx]
                    dynamics_with_filtered_time = eqx.tree_at(
                        lambda m: m.t0,
                        dynamics,
                        filtered_time,
                        is_leaf=lambda x: x is None,
                    )
                    dynamics_with_filtered_ic = eqx.tree_at(
                        lambda m: m.initial_condition,
                        dynamics_with_filtered_time,
                        filtered_dist,
                        is_leaf=lambda x: x is None,
                    )

                    mask_seg = predict_times >= filtered_time
                    if f_idx + 1 < len(filtered_times):
                        mask_seg = mask_seg & (
                            predict_times < filtered_times[f_idx + 1]
                        )
                    sub = _extract_times(mask_seg, empty_fill=filtered_time)
                    seg_masks.append(mask_seg)

                    ctrl_t_seg, ctrl_v_seg = _ctrl_for_segment(sub)
                    sim_results.append(
                        self._simulate(
                            f"{name}_{f_idx + 1}",
                            dynamics_with_filtered_ic,
                            obs_times=None,
                            obs_values=None,
                            ctrl_times=ctrl_t_seg,
                            ctrl_values=ctrl_v_seg,
                            predict_times=sub,
                        )
                    )

            # Merge segment results into predict_times order.
            # Each segment has fixed-size output (n_pred) with valid entries first, padded.
            # Use seg_masks to scatter each segment's values into the correct positions.
            states_list = [r["states"] for r in sim_results]
            obs_list = [r["observations"] for r in sim_results]

            def _merge_segments(arr_list, seg_masks):
                """Merge per-segment arrays into a single output array.

                At each position ``i`` in the merged output, the value is taken
                from the unique segment whose mask has ``mask[i] == True``.

                All arrays must have shape ``(n_sim, T_seg, state_dim)`` — the
                ndim==3 contract guaranteed by ``_ensure_trailing_dim``.
                """
                first = arr_list[0]
                assert first.ndim == 3, (
                    f"_merge_segments expects ndim==3 arrays (n_sim, T, D), got ndim={first.ndim} "
                    f"with shape {first.shape}. Ensure _ensure_trailing_dim is applied before "
                    "calling this function."
                )
                out = jnp.zeros(
                    (first.shape[0], n_pred, first.shape[2]), dtype=first.dtype
                )
                for arr, mask in zip(arr_list, seg_masks):
                    cumsum = jnp.cumsum(mask)
                    local_idx = jnp.where(mask, cumsum - 1, 0)
                    gathered = arr[:, local_idx, :]
                    mask_bc = jnp.expand_dims(jnp.expand_dims(mask, 0), -1)  # (1, T, 1)
                    out = jnp.where(mask_bc, gathered, out)
                return out

            sim_results_dict = {
                "predicted_states": _merge_segments(states_list, seg_masks),
                "predicted_observations": _merge_segments(obs_list, seg_masks),
            }
            n_sim_out = sim_results_dict["predicted_states"].shape[0]
            sim_results_dict["predicted_times"] = _tile_times(predict_times, n_sim_out)

        else:
            sim_results_dict = self._simulate(
                name,
                dynamics,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                predict_times=predict_times,
                **kwargs,
            )

        # Add the results from the simulator as deterministic sites
        for site_name, trajectory in sim_results_dict.items():
            numpyro.deterministic(f"{name}_{site_name}", trajectory)

        return fwd(
            name,
            dynamics,
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
    """Solve DE (ODE or SDE) with a single diffeqsolve call.

    Branches on diffusion_coefficient: None -> ODE, else -> SDE.
    t0 is explicit (may differ from model's t0, e.g. for predict_times from filter).
    """
    t1 = saveat_times[-1]

    # Use lax.cond to avoid TracerBoolConversionError when t0/t1 are traced
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
    """Emit observations. ODE: numpyro.sample with obs=. SDE/vmap: dist.sample(key).

    When key is None (ODE n_sim=1 path), uses numpyro.sample and supports obs=
    conditioning. Returns array of shape (T, obs_dim).

    When key is not None (SDE or ODE n_sim>1 vmap path), samples from dist directly;
    obs_values must be None (callers enforce this). Returns array of shape (T, obs_dim).
    """
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

        if predict_times is None:
            warnings.warn(
                "predict_times is not provided to an SDESimulator; SDESimulator will simply return its inputs."
            )
            # TODO: Handle this case.
            raise NotImplementedError(
                "this is to-be-implemented. Should pass forward whatever is from previous operator in **kwargs."
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

        def _run_one_from_x0(key: Array, x0: Array) -> tuple[Array, Array]:
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
        if n_sim == 1:
            with numpyro.plate(f"{name}_n_simulations", 1):
                initial_state = numpyro.sample(
                    f"{name}_x_0", dynamics.initial_condition
                )
            initial_state_arr = cast(Array, jnp.asarray(initial_state))
            states, emissions = _run_one_from_x0(prng_key, initial_state_arr[0])
            # Always return (n_sim, T, ...) for consistent shaping
            states = jnp.expand_dims(states, axis=0)
            emissions = jnp.expand_dims(emissions, axis=0)
            return {
                "times": _tile_times(times, 1),
                "states": states,
                "observations": emissions,
            }

        with numpyro.plate(f"{name}_n_simulations", n_sim):
            initial_state = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        keys = jr.split(prng_key, n_sim)
        states, emissions = jax.vmap(_run_one_from_x0)(keys, jnp.asarray(initial_state))
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
        if n_sim > 1 and obs_values is not None:
            raise ValueError(
                "n_simulations > 1 is only supported when obs_values is None (forward simulation)"
            )

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

        # n_simulations > 1: vmap over scan with dist.sample (no numpyro.sample in body)
        if n_sim > 1:
            with numpyro.plate(f"{name}_n_simulations", n_sim):
                initial_state = numpyro.sample(
                    f"{name}_x_0", dynamics.initial_condition
                )
            prng_key = numpyro.prng_key()
            if prng_key is None:
                raise ValueError("PRNG key required for n_simulations > 1")
            keys = jr.split(prng_key, n_sim)

            def _run_one(key, x0):
                keys_t = jr.split(key, T)

                def _step(carry, t_idx):
                    x_prev = carry
                    k_trans, k_obs = jr.split(keys_t[t_idx], 2)
                    t_now = times[t_idx]
                    t_next = times[t_idx + 1]
                    u_now = _get_val_or_None(ctrl_values, t_idx)
                    u_next = _get_val_or_None(ctrl_values, t_idx + 1)
                    trans = dynamics.state_evolution(
                        x=x_prev, u=u_now, t_now=t_now, t_next=t_next
                    )
                    x_t = trans.sample(k_trans)
                    obs_dist = dynamics.observation_model(x=x_t, u=u_next, t=t_next)
                    y_t = obs_dist.sample(k_obs)
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

            states, observations = jax.vmap(_run_one)(keys, initial_state)
            return {
                "times": _tile_times(times, n_sim),
                "states": _ensure_trailing_dim(states),
                "observations": _ensure_trailing_dim(observations),
            }

        # Default: scan over time (n_simulations == 1)
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
            t_now = times[t_idx]
            t_next = times[t_idx + 1]
            u_now = _get_val_or_None(ctrl_values, t_idx)
            u_next = _get_val_or_None(ctrl_values, t_idx + 1)
            with numpyro.plate(f"{name}_n_simulations", 1):
                x_t_site = numpyro.sample(
                    f"{name}_x_{t_idx + 1}",
                    dynamics.state_evolution(
                        x=x_prev, u=u_now, t_now=t_now, t_next=t_next
                    ),
                )
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
        if n_sim > 1 and obs_values is not None:
            raise ValueError(
                "n_simulations > 1 is only supported when obs_values is None (forward simulation)"
            )

        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, times)
            control_path_eval = lambda t: control_path.evaluate(t, left=False)
        else:
            control_path_eval = lambda t: None

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]

        def _run_one(x0: Array, *, obs_key=None):
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

        if n_sim == 1:
            with numpyro.plate(f"{name}_n_simulations", 1):
                x0 = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
            x0_site_arr = cast(Array, jnp.asarray(x0))
            x0_arr: Array = x0_site_arr[0]
            states, observations = _run_one(x0_arr)
            # Always return (n_sim, T, ...) for consistent shaping
            return {
                "times": _tile_times(times, 1),
                "states": jnp.expand_dims(states, axis=0),
                "observations": jnp.expand_dims(observations, axis=0),
            }

        with numpyro.plate(f"{name}_n_simulations", n_sim):
            initial_state = numpyro.sample(f"{name}_x_0", dynamics.initial_condition)
        prng_key = numpyro.prng_key()
        if prng_key is None:
            raise ValueError("PRNG key required for n_simulations > 1")
        obs_keys = jr.split(prng_key, n_sim)
        states, observations = jax.vmap(_run_one)(
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
