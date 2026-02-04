import dataclasses
import warnings

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from jax import Array
from jax.tree_util import tree_map
from numpyro.contrib.control_flow import scan as nscan

from dynestyx.dynamical_models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
    State,
)
from dynestyx.handlers import BaseSimulator
from dynestyx.observations import DiracIdentityObservation
from dynestyx.ops import Context, States
from dynestyx.utils import (
    _get_controls,
    _get_val_or_None,
    _validate_control_dim,
    diffeqsolve_util,
    infer_batch_shape,
)

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


class SDESimulator(BaseSimulator):
    """Simulator that works with ContinuousTimeStateEvolution with stochastic dynamics."""

    def __init__(
        self,
        key: Array,
        solver: dfx.AbstractSolver = dfx.Heun(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        dt0: float = 0.01,
        tol_vbt: float = 1e-1,  # tolerance for virtual brownian tree
        max_steps: int = int(1e5),
    ):
        self.key = key  # key for model randomness (initial condition, SDE solver, and observation noise)
        self.diffeqsolve_settings = {
            "solver": solver,
            "stepsize_controller": stepsize_controller,
            "adjoint": adjoint,
            "dt0": dt0,
            "tol_vbt": tol_vbt,
            "max_steps": max_steps,
        }

    def simulate(self, context: Context, dynamics) -> States:
        if not isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            raise NotImplementedError(
                f"SDESimulator only works with ContinuousTimeStateEvolution, got {type(dynamics.state_evolution)}"
            )

        if (
            dynamics.state_evolution.diffusion_coefficient is None
            or dynamics.state_evolution.diffusion_covariance is None
        ):
            raise ValueError(
                "SDESimulator requires both diffusion_coefficient and diffusion_covariance to be "
                f"defined (got coeff={dynamics.state_evolution.diffusion_coefficient}, "
                f"cov={dynamics.state_evolution.diffusion_covariance}). "
                "Use ODESimulator for deterministic dynamics."
            )

        # Extract times from context
        if context.observations is None or context.observations.times is None:
            raise ValueError("context.observations.times must be provided")
        raw_times = jnp.asarray(context.observations.times)

        # Detect batch shape from surrounding plates (if any)
        batch_shape = infer_batch_shape() or ()
        batch_size = int(jnp.prod(jnp.array(batch_shape))) if batch_shape else 0

        def _ensure_batch_prefix(x):
            if x is None or not batch_shape:
                return x
            if tuple(x.shape[: len(batch_shape)]) == tuple(batch_shape):
                return x
            return jnp.broadcast_to(x, batch_shape + x.shape)

        def _flatten_batch_prefix(x):
            if x is None or not batch_shape:
                return x
            if tuple(x.shape[: len(batch_shape)]) != tuple(batch_shape):
                return x
            return x.reshape((batch_size,) + x.shape[len(batch_shape) :])

        def _drift_has_batched_params(drift_obj) -> bool:
            if not batch_shape:
                return False
            for leaf in jax.tree.leaves(drift_obj):
                if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1:
                    if leaf.shape[0] == batch_size:
                        return True
            return False

        def _extract_time_vector(ts: Array) -> Array:
            """Return a 1D time vector of shape (T,) from any array shaped (..., T)."""
            if ts.ndim == 1:
                return ts
            # Treat the last axis as time and pick the first leading index.
            return ts.reshape((-1, ts.shape[-1]))[0]

        def _extract_control_trajectory(us: Array) -> Array:
            """
            Return a control trajectory shaped (T, control_dim) from any array shaped (..., T, control_dim).
            """
            if us.ndim == 2:
                return us
            return us.reshape((-1,) + us.shape[-2:])[0]

        def _move_control_time_first(us: Array) -> Array:
            # Controls are assumed shaped (..., T, control_dim) after extraction/broadcast.
            return jnp.moveaxis(us, -2, 0)

        with numpyro.handlers.seed(rng_seed=self.key):
            # Extract controls from context if available
            times = _extract_time_vector(raw_times)
            ctrl_times, ctrl_values = _get_controls(context, times)

            # Validate that control_dim is set when controls are present
            _validate_control_dim(dynamics, ctrl_values)

            # Diffrax expects scalar times; keep the integration time grid unbatched.
            # (If callers provide batched times, we pick the first time grid.)
            times_scan = times

            initial_state = numpyro.sample("x_0", dynamics.initial_condition)
            initial_state = _ensure_batch_prefix(initial_state)

            if ctrl_values is not None:
                ctrl_values = _extract_control_trajectory(ctrl_values)
                ctrl_values = _ensure_batch_prefix(ctrl_values)
                ctrl_scan = _move_control_time_first(
                    ctrl_values
                )  # (T, ..., control_dim)
                u_0 = ctrl_scan[0]
            else:
                ctrl_scan = None
                u_0 = None

            y_0 = numpyro.sample(
                "y_0",
                dynamics.observation_model(x=initial_state, u=u_0, t=times_scan[0]),
            )
            y_0 = _ensure_batch_prefix(y_0)

            def _step(prev_state, args):
                t_next_idx, t0, t1, inpt = args

                numpyro_key = numpyro.prng_key()

                def integrate_once(key, y0, t0_single, t1_single, inpt_single):
                    def drift(t, y, args):
                        return dynamics.state_evolution.drift(y, inpt_single, t)

                    def diffusion(t, y, args):
                        Qc_t = dynamics.state_evolution.diffusion_covariance(
                            y, inpt_single, t
                        )
                        L_t = dynamics.state_evolution.diffusion_coefficient(
                            y, inpt_single, t
                        )

                        Q_sqrt = jnp.linalg.cholesky(Qc_t)
                        combined = L_t @ Q_sqrt
                        # If the model provides unbatched diffusion (state_dim,state_dim)
                        # but the state is batched (...,state_dim), broadcast across the
                        # leading batch dims so diffrax can contract with the control.
                        if y.ndim > 1 and combined.ndim == 2:
                            combined = jnp.broadcast_to(
                                combined, y.shape[:-1] + combined.shape
                            )
                        return combined

                    # If times were broadcast, pick a representative scalar
                    t0_scalar = t0
                    t1_scalar = t1

                    return diffeqsolve_util(
                        key=key,
                        drift=drift,
                        diffusion=diffusion,
                        t0=t0_scalar,
                        t1=t1_scalar,
                        y0=y0,
                        **self.diffeqsolve_settings,
                    )[0]

                if batch_shape:
                    # If the drift is a PyTree (e.g. eqx.Module) with batched parameter leaves,
                    # then vmap over the batch and slice those leaves automatically.
                    drift_obj = dynamics.state_evolution.drift
                    if _drift_has_batched_params(drift_obj):
                        keys = jr.split(numpyro_key, batch_size)
                        prev_state = _flatten_batch_prefix(prev_state)
                        inpt_flat = (
                            _flatten_batch_prefix(inpt) if inpt is not None else None
                        )

                        def integrate_single(drift_i, key_i, y0_i, inpt_i):
                            def drift(t, y, args):
                                return drift_i(y, inpt_i, t)

                            def diffusion(t, y, args):
                                Qc_t = dynamics.state_evolution.diffusion_covariance(
                                    y, inpt_i, t
                                )
                                L_t = dynamics.state_evolution.diffusion_coefficient(
                                    y, inpt_i, t
                                )
                                Q_sqrt = jnp.linalg.cholesky(Qc_t)
                                return L_t @ Q_sqrt

                            return diffeqsolve_util(
                                key=key_i,
                                drift=drift,
                                diffusion=diffusion,
                                t0=t0,
                                t1=t1,
                                y0=y0_i,
                                **self.diffeqsolve_settings,
                            )[0]

                        if inpt_flat is None:
                            state = jax.vmap(
                                lambda drift_i, key_i, y0_i: integrate_single(
                                    drift_i, key_i, y0_i, None
                                ),
                                in_axes=(0, 0, 0),
                            )(drift_obj, keys, prev_state)
                        else:
                            state = jax.vmap(integrate_single, in_axes=(0, 0, 0, 0))(
                                drift_obj, keys, prev_state, inpt_flat
                            )
                        state = state.reshape(batch_shape + state.shape[1:])
                    else:
                        state = integrate_once(numpyro_key, prev_state, t0, t1, inpt)
                else:
                    state = integrate_once(numpyro_key, prev_state, t0, t1, inpt)

                state = numpyro.deterministic(f"x_{t_next_idx}", state)

                emission = numpyro.sample(
                    f"y_{t_next_idx}",
                    dynamics.observation_model(x=state, u=inpt, t=t1),
                )

                return state, (state, emission)

            num_timesteps = times_scan.shape[0]
            t0 = times_scan[:-1]
            t1 = times_scan[1:]
            next_inputs = ctrl_scan[1:] if ctrl_scan is not None else None

            _, (next_states, next_emissions) = nscan(
                _step,
                initial_state,
                (jnp.arange(1, num_timesteps), t0, t1, next_inputs),
            )

        def expand_and_cat(x0, x1T):
            return jnp.concatenate((jnp.expand_dims(x0, 0), x1T))

        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, y_0, next_emissions)

        return {"times": times, "states": states, "observations": emissions}

    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Extract observed trajectory (original 1D times)
        obs_traj = context.observations
        if obs_traj is None or obs_traj.times is None:
            raise ValueError("context.observations.times must be provided")
        obs_times = obs_traj.times
        obs_values = obs_traj.values if obs_traj is not None else None
        if isinstance(obs_values, dict):
            raise ValueError("obs_values must be an Array or None, not a dict")

        # Controls aligned with observed times
        _, ctrl_values = _get_controls(context, obs_times)

        # Run the simulator to obtain states/emissions (times reshaped as needed for cd-dynamax)
        simulated = self.simulate(context, dynamics)
        states = simulated["states"]
        emissions = simulated["observations"]

        # Deterministic sites for times, states, emissions
        numpyro.deterministic("times", obs_times)
        numpyro.deterministic("states", states)
        numpyro.deterministic("observations", emissions)

        # Observation sample sites using the model's observation distribution only when obs values are provided.
        # If obs_values is None and emissions were produced, we avoid resampling to keep consistency with cd-dynamax.
        if obs_values is not None:
            T = len(obs_times)
            warnings.warn(
                "Adding observation sample sites in the SDESimulator will not result in proper state inference. While it provides a technically unbiased estimate of the marginal likelihood for system identification, it is highly inefficient and is not recommended."
            )
            for t_idx in range(T):
                u_t = _get_val_or_None(ctrl_values, t_idx)
                t = obs_times[t_idx]
                numpyro.sample(
                    f"y_{t_idx}",
                    dynamics.observation_model(states[t_idx], u_t, t),
                    obs=_get_val_or_None(obs_values, t_idx),
                )


@dataclasses.dataclass
class DiscreteTimeSimulator(BaseSimulator):
    """Simulator for discrete-time dynamical models.

    Assumes we have ic, transition, and observation distributions,
    as well as (time_index, observation) pairs in the context.

    Simply unrolls the model and adds obs=data as you would in numpyro.

    This does not explicitly add logfactors; it lets numpyro do it automatically.
    Instead, it just unrolls the model and adds observed sites
    (which numpyro uses to compute logfactors).
    """

    def simulate(
        self,
        context: Context,
        dynamics: DynamicalModel,
    ) -> States:
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")
        if isinstance(obs_traj.values, dict):
            raise ValueError("obs_traj.values must be an Array or None, not a dict")
        obs_values = obs_traj.values

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs_times)

        T = len(obs_times)

        # DiracIdentityObservation with observed values: y_t = x_t, so we use plating
        # instead of scan. state_evolution returns a dist; call it with batched inputs.
        if isinstance(dynamics.observation_model, DiracIdentityObservation) and (
            obs_values is not None
        ):
            numpyro.sample("x_0", dynamics.initial_condition, obs=obs_values[0])
            numpyro.deterministic("y_0", obs_values[0])

            x_prev = obs_values[:-1]
            x_next = obs_values[1:]
            u_prev = ctrl_values[:-1] if ctrl_values is not None else None
            t_now = obs_times[:-1]
            t_next = obs_times[1:]

            with numpyro.plate("time", T - 1):
                trans = dynamics.state_evolution(
                    x_prev,
                    u_prev,
                    t_now,
                    t_next,  # type: ignore
                )
                numpyro.sample("x_next", trans, obs=x_next)  # type: ignore

            return {
                "times": obs_times,
                "states": obs_values,
                "observations": obs_values,
            }

        # Default: scan over time
        # Sample initial state
        x_prev: State = numpyro.sample("x_0", dynamics.initial_condition)  # type: ignore

        # sample initial observation
        u_0 = _get_val_or_None(ctrl_values, 0)
        y_0 = numpyro.sample(
            "y_0",
            dynamics.observation_model(x_prev, u_0, obs_times[0]),
            obs=_get_val_or_None(obs_values, 0),
        )

        def _step(x_prev, t_idx):
            t_now = obs_times[t_idx]
            t_next = obs_times[t_idx + 1]
            u_now = _get_val_or_None(ctrl_values, t_idx)
            u_next = _get_val_or_None(ctrl_values, t_idx + 1)
            # Sample next state
            x_t = numpyro.sample(
                f"x_{t_idx + 1}",
                dynamics.state_evolution(x=x_prev, u=u_now, t_now=t_now, t_next=t_next),
            )

            # Sample observation
            y_t = numpyro.sample(
                f"y_{t_idx + 1}",
                dynamics.observation_model(x=x_t, u=u_next, t=t_next),
                obs=_get_val_or_None(obs_values, t_idx + 1),
            )
            return x_t, (x_t, y_t)

        # Run scan and collect states and observations
        # scan_outputs will be (scan_states, scan_observations) where each is shape (T-1, ...)
        _, scan_outputs = nscan(_step, x_prev, jnp.arange(T - 1))
        scan_states, scan_observations = scan_outputs

        # Stack initial state/observation with scanned results
        # x_prev is shape (state_dim,) or scalar, scan_states is (T-1, state_dim)
        # y_0 is shape (obs_dim,) or scalar, scan_observations is (T-1, obs_dim)
        # Use expand_dims to ensure proper shape for concatenation
        # shape (1, state_dim) or (1,)
        x_0_expanded = jnp.expand_dims(x_prev, axis=0)  # type: ignore
        y_0_expanded = jnp.expand_dims(y_0, axis=0)  # shape (1, obs_dim) or (1,)
        states = jnp.concatenate(
            [x_0_expanded, scan_states], axis=0
        )  # shape (T, state_dim)
        observations = jnp.concatenate(
            [y_0_expanded, scan_observations], axis=0
        )  # shape (T, obs_dim)

        return {"times": obs_times, "states": states, "observations": observations}


@dataclasses.dataclass
class ODESimulator(BaseSimulator):
    """Simulator for continuous-time deterministic (ODE) dynamical models.

    Assumes we have ic, transition, and observation distributions,
    as well as (time_index, observation) pairs in the context.

    Simply unrolls the model and adds obs=data as you would in numpyro.

    This does not add logfactors, just unrolls the model and adds observed sites.
    """

    solver: dfx.AbstractSolver = dfx.Tsit5()
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize()
    dt0: float = 0.01
    max_steps: int = 10_000

    def simulate(
        self,
        context: Context,
        dynamics: DynamicalModel,
    ) -> States:
        # Pull observed trajectory from context
        obs_traj = context.observations
        obs_times = obs_traj.times
        obs_values = obs_traj.values
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")
        if isinstance(obs_values, dict):
            raise ValueError("obs_values must be an Array or None, not a dict")

        # Pull control trajectory from context and validate
        ctrl_times, ctrl_values = _get_controls(context, obs_times)

        T = len(obs_times)

        # Sample initial state
        x_prev = numpyro.sample("x_0", dynamics.initial_condition)

        # Create drift function that interpolates controls
        # For now, use piecewise constant (nearest neighbor) interpolation
        if ctrl_times is not None and ctrl_values is not None:
            # Create LinearInterpolation for controls using diffrax
            control_path = dfx.LinearInterpolation(ts=ctrl_times, ys=ctrl_values)

            def f(t, y, args):
                # Evaluate control at time t using interpolation
                u_t = control_path.evaluate(t)
                return dynamics.state_evolution.drift(x=y, u=u_t, t=t)
        else:

            def f(t, y, args):
                return dynamics.state_evolution.drift(x=y, u=None, t=t)

        # Solve ODE at all observation times using diffrax
        sol = dfx.diffeqsolve(
            terms=dfx.ODETerm(f),
            solver=self.solver,
            t0=obs_times[0],
            t1=obs_times[-1],
            dt0=self.dt0,
            y0=x_prev,
            saveat=dfx.SaveAt(ts=obs_times),
            stepsize_controller=self.stepsize_controller,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )
        x_sol = sol.ys  # shape (T, state_dim) # includes initial state at t0

        # use scan to sample observations and collect them
        def _step(carry, t_idx):
            x_t = x_sol[t_idx]
            t = obs_times[t_idx]
            u_t = _get_val_or_None(ctrl_values, t_idx)
            # Sample observation
            y_t = numpyro.sample(
                f"y_{t_idx}",
                dynamics.observation_model(x=x_t, u=u_t, t=t),
                obs=_get_val_or_None(obs_values, t_idx),
            )
            return carry, y_t

        # Run scan and collect observations
        # scan_outputs will be observations of shape (T, obs_dim)
        _, scan_observations = nscan(_step, None, jnp.arange(T))
        observations = scan_observations  # shape (T, obs_dim)

        return {"times": obs_times, "states": x_sol, "observations": observations}
