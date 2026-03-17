"""NumPyro-aware simulators/unrollers for dynamical models."""

import dataclasses
from collections.abc import Callable

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
import numpyro
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from jax import Array
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
    _validate_control_dim,
    _validate_controls,
)

def _sample_with_nan_mask(name, dist, obs_values, idx):
    """Sample from ``dist``, conditioning on ``obs_values[idx]``.

    Handles entirely-missing observations (all-NaN rows) via
    ``numpyro.factor``.  When observed, the factor corrects the
    log-density so the effective contribution equals
    ``dist.log_prob(obs)``.  When missing, the sample is drawn freely
    from ``dist``.

    If ``obs_values`` is None, samples unconditionally.

    Note: partial missingness (some but not all components NaN in a row) must
    be caught *before* calling this function.
    """
    if obs_values is None:
        return numpyro.sample(name, dist)
    obs = obs_values[idx]
    is_missing = jnp.isnan(obs).all()
    obs_filled = jnp.where(jnp.isnan(obs), 0.0, obs)
    # Always sample from the distribution (latent site).
    x_sampled = numpyro.sample(name, dist)
    # Correct the log-density: observed rows get log_prob(obs) instead
    # of log_prob(x_sampled); missing rows get no correction (0).
    numpyro.factor(
        f"{name}_obs",
        jnp.where(
            is_missing,
            0.0,
            dist.log_prob(obs_filled) - dist.log_prob(x_sampled),
        ),
    )
    return jnp.where(is_missing, x_sampled, obs_filled)


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
        **kwargs,
    ) -> FunctionOfTime:
        self._add_solved_sites(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
        return fwd(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

    def _add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ):
        # Only simulate if we have observation times
        if obs_times is None:
            return

        # Run the simulator
        simulated = self._simulate(
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

        # Add the results from the simulator as deterministic sites
        for site_name, trajectory in simulated.items():
            numpyro.deterministic(site_name, trajectory)

    def _simulate(
        self,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
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

        Returns:
            dict[str, State]: Mapping from deterministic site names to
                trajectories. Conventionally includes `"times"`, `"states"`,
                and `"observations"`.
        """
        raise NotImplementedError()


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

        if tol_vbt is None:
            self.tol_vbt = dt0 / 2.0
        else:
            self.tol_vbt = tol_vbt

        assert self.tol_vbt < dt0, (
            "tol_vbt must be smaller than dt0 for statistically correct simulation."
        )

    def _simulate(
        self,
        dynamics,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
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

        if obs_times is None:
            raise ValueError("obs_times must be provided")
        times = obs_times

        _validate_controls(times, ctrl_times, ctrl_values)
        _validate_control_dim(dynamics, ctrl_values)

        initial_state = numpyro.sample("x_0", dynamics.initial_condition)

        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, times)
            control_path_eval: Callable[[Array], Array | None] = lambda t: (
                control_path.evaluate(t, left=False)
            )
        else:
            control_path_eval = lambda t: None

        def _drift(t, y, args):
            u_t = args(t)
            return dynamics.state_evolution.total_drift(x=y, u=u_t, t=t)

        def _diffusion(t, y, args):
            u_t = args(t)
            return dynamics.state_evolution.diffusion_coefficient(x=y, u=u_t, t=t)

        if dynamics.state_evolution.bm_dim is None:
            raise ValueError(
                "SDESimulator requires state_evolution.bm_dim to be inferred. "
                "Construct dynamics via DynamicalModel before simulation."
            )

        bm = dfx.VirtualBrownianTree(
            t0=times[0],
            t1=times[-1],
            tol=self.tol_vbt,
            shape=(dynamics.state_evolution.bm_dim,),
            key=numpyro.prng_key(),
        )

        terms = dfx.MultiTerm(  # type: ignore
            dfx.ODETerm(_drift), dfx.ControlTerm(_diffusion, bm)
        )

        sol = dfx.diffeqsolve(
            terms,
            t0=times[0],
            t1=times[-1],
            y0=initial_state,
            args=control_path_eval,
            saveat=dfx.SaveAt(ts=times),
            **self.diffeqsolve_settings,
        )
        states_sol = sol.ys  # (T, ..., state_dim)

        def _create_observations_step(carry, t_idx):
            x_t = states_sol[t_idx]
            t = times[t_idx]
            u_t = control_path_eval(t)
            y_t = _sample_with_nan_mask(
                f"y_{t_idx}",
                dynamics.observation_model(x=x_t, u=u_t, t=t),
                obs_values,
                t_idx,
            )
            return carry, y_t

        states = states_sol
        _, emissions = nscan(_create_observations_step, None, jnp.arange(len(times)))

        return {"times": times, "states": states, "observations": emissions}


@dataclasses.dataclass
class DiscreteTimeSimulator(BaseSimulator):
    """Simulator for discrete-time dynamical models.

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

    def _simulate(
        self,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
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

        Returns:
            dict[str, State]: Dictionary with `"times"`, `"states"`, and
                `"observations"` trajectories.
        """
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")

        _validate_controls(obs_times, ctrl_times, ctrl_values)

        T = len(obs_times)
        if T < 1:
            raise ValueError("obs_times must contain at least one timepoint")

        # DiracIdentityObservation with observed values: y_t = x_t, so we use plating
        # instead of scan. state_evolution returns a dist; call it with batched inputs.
        #
        # When there are missing rows (entire-row NaN), we filter them out
        # using numpy (concrete indexing, no tracers) before entering the
        # plate.  state_evolution handles non-unit dt from skipped rows.
        has_no_obs = obs_values is None
        has_missing_data = (
            not has_no_obs and np.isnan(np.asarray(obs_values)).any()
        )

        if has_missing_data:
            # Only entire-row missingness is supported; raise on partial.
            obs_np = np.asarray(obs_values)
            nan_per_row = np.isnan(obs_np).any(axis=1)
            all_nan_per_row = np.isnan(obs_np).all(axis=1)
            has_partial = (nan_per_row & ~all_nan_per_row).any()
            if has_partial:
                raise ValueError(
                    "Partial missingness (some but not all components NaN in a "
                    "row) is not yet supported. Only entire-row NaN is allowed."
                )

        if isinstance(dynamics.observation_model, DiracIdentityObservation) and not has_no_obs:
            # When missing rows exist, filter to observed rows using numpy
            # (concrete, no tracers).  state_evolution handles non-unit dt
            # from skipped rows.  When no missing data, use arrays as-is.
            if has_missing_data:
                obs_np = np.asarray(obs_values)
                observed_mask = ~np.isnan(obs_np).all(axis=1)
                obs_values = jnp.array(obs_np[observed_mask])
                obs_times = jnp.array(np.asarray(obs_times)[observed_mask])
                if ctrl_values is not None:
                    ctrl_values = jnp.array(np.asarray(ctrl_values)[observed_mask])
                T = len(obs_times)

            numpyro.sample("x_0", dynamics.initial_condition, obs=obs_values[0])
            numpyro.deterministic("y_0", obs_values[0])
            if T == 1:
                # No transitions exist for a single-timepoint trajectory.
                return {
                    "times": obs_times,
                    "states": obs_values,
                    "observations": obs_values,
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
            t_now = obs_times[:-1]
            t_next = obs_times[1:]

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
        y_0 = _sample_with_nan_mask(
            "y_0",
            dynamics.observation_model(x_prev, u_0, obs_times[0]),
            obs_values,
            0,
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

            # Sample observation (with NaN masking for missing rows)
            y_t = _sample_with_nan_mask(
                f"y_{t_idx + 1}",
                dynamics.observation_model(x=x_t, u=u_next, t=t_next),
                obs_values,
                t_idx + 1,
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
    """Simulator for continuous-time deterministic dynamics (ODEs).

    This unrolls a `ContinuousTimeStateEvolution` with **no diffusion** by solving
    an ODE using Diffrax and then emitting observations at `obs_times` as NumPyro
    sample sites. Solver options can be configured via the constructor.

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

    solver: dfx.AbstractSolver = dfx.Tsit5()
    adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint()
    stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize()
    dt0: float = 1e-3
    max_steps: int = 10_000

    def __init__(
        self,
        solver: dfx.AbstractSolver = dfx.Tsit5(),
        adjoint: dfx.AbstractAdjoint = dfx.RecursiveCheckpointAdjoint(),
        stepsize_controller: dfx.AbstractStepSizeController = dfx.ConstantStepSize(),
        dt0: float = 1e-3,
        max_steps: int = 100_000,
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
        """
        self.solver = solver
        self.adjoint = adjoint
        self.stepsize_controller = stepsize_controller
        self.dt0 = dt0
        self.max_steps = max_steps

    def _simulate(
        self,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> dict[str, State]:
        """Unroll a deterministic continuous-time model as a NumPyro model.

        This method:
        - samples the initial state as `numpyro.sample("x_0", ...)`,
        - solves the ODE and saves the solution at `obs_times`,
        - emits observations as `numpyro.sample("y_i", ..., obs=...)`.

        Args:
            dynamics: A `DynamicalModel` whose `state_evolution` is a
                `ContinuousTimeStateEvolution` with deterministic dynamics.
            obs_times: Times at which to save the latent state and emit observations.
                Required.
            obs_values: Optional observation array. If provided, observation sites are
                conditioned via `obs=obs_values[i]`.
            ctrl_times: Optional control times.
            ctrl_values: Optional controls aligned to `ctrl_times`.

        Returns:
            dict[str, State]: Dictionary with `"times"`, `"states"`, and
                `"observations"` trajectories.
        """
        if obs_times is None:
            raise ValueError("obs_times must be provided, but got None")

        _validate_controls(obs_times, ctrl_times, ctrl_values)

        T = len(obs_times)

        # Sample initial state
        x_prev = numpyro.sample("x_0", dynamics.initial_condition)

        # Create drift function that interpolates controls
        if ctrl_times is not None and ctrl_values is not None:
            control_path = _build_control_path(ctrl_times, ctrl_values, obs_times)

            def f(t, y, args):
                # Evaluate control at time t using interpolation
                u_t = args(t)
                return dynamics.state_evolution.total_drift(x=y, u=u_t, t=t)

            args = lambda t: control_path.evaluate(t, left=False)

        else:

            def f(t, y, args):
                return dynamics.state_evolution.total_drift(x=y, u=None, t=t)

            args = None

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
            args=args,
        )
        x_sol = sol.ys  # shape (T, state_dim) # includes initial state at t0

        # use scan to sample observations and collect them
        def _step(carry, t_idx):
            x_t = x_sol[t_idx]
            t = obs_times[t_idx]
            u_t = None if args is None else args(t)
            # Sample observation (with NaN masking for missing rows)
            y_t = _sample_with_nan_mask(
                f"y_{t_idx}",
                dynamics.observation_model(x=x_t, u=u_t, t=t),
                obs_values,
                t_idx,
            )
            return carry, y_t

        # Run scan and collect observations
        # scan_outputs will be observations of shape (T, obs_dim)
        _, scan_observations = nscan(_step, None, jnp.arange(T))
        observations = scan_observations  # shape (T, obs_dim)

        return {"times": obs_times, "states": x_sol, "observations": observations}


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
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

        self.simulator = None

    def _simulate(
        self,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> dict[str, State]:
        if self.simulator is None:
            raise ValueError("Simulator not initialized. This shouldn't happen.")

        return self.simulator._simulate(
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

    def _add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ):
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

        return self.simulator._add_solved_sites(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
