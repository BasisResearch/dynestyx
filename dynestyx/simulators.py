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
    ObservationModel,
)
from dynestyx.models.core import WithPartialMissingnessSupport
from dynestyx.types import FunctionOfTime, State
from dynestyx.utils import (
    _build_control_path,
    _get_val_or_None,
    _validate_control_dim,
    _validate_controls,
)


def _per_dim_log_prob(d, y):
    """Extract per-element log_prob (shape = event_shape) from a distribution.

    Supports:
      - Independent(base, 1): returns base.log_prob(y), shape (event_dim,)
      - MultivariateNormal with diagonal covariance: decomposes into Normal per dim

    Raises:
        NotImplementedError: for non-decomposable distributions.
    """
    import numpyro.distributions as _nd

    if isinstance(d, _nd.Normal):
        return d.log_prob(y)  # scalar for scalar state
    if isinstance(d, _nd.Independent) and d.reinterpreted_batch_ndims == 1:
        return d.base_dist.log_prob(y)
    if isinstance(d, _nd.MultivariateNormal):
        # Use diagonal(..., axis1=-2, axis2=-1) so it works for both plain
        # (n, n) and batched (..., n, n) covariance matrices.
        diag_var = jnp.diagonal(d.covariance_matrix, axis1=-2, axis2=-1)
        std = jnp.sqrt(diag_var)
        return _nd.Normal(d.loc, std).log_prob(y)
    raise NotImplementedError(
        f"_per_dim_log_prob not implemented for {type(d).__name__}. "
        "Requires Independent(Normal, 1) or MultivariateNormal with diagonal covariance."
    )


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
            y_t = numpyro.sample(
                f"y_{t_idx}",
                dynamics.observation_model(x=x_t, u=u_t, t=t),
                obs=_get_val_or_None(obs_values, t_idx),
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

    Args:
        unroll_missing: When True and obs_values contains entirely-missing rows,
            step through all T time steps (sampling latent states for missing rows)
            rather than filtering them out. Produces full-length output arrays with
            NaN in the observations at missing rows. Default False preserves the
            existing skip/filter behavior.
    """

    unroll_missing: bool = False

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

        has_no_obs = obs_values is None
        is_dirac = isinstance(dynamics.observation_model, DiracIdentityObservation)
        has_missing_data = not has_no_obs and np.isnan(np.asarray(obs_values)).any()

        use_two_mask_scan = False
        obs_mask_jax = None
        safe_obs_jax = None
        latent_mask_jax = None

        # Determine whether to use the two-mask scan path.
        # Triggered by:
        #   (a) partial missingness (some but not all dims NaN in a row), OR
        #   (b) unroll_missing=True with any missing data or DiracIdentity obs.
        # Otherwise use the plate path (DiracIdentity) or existing filter path.
        if not has_no_obs and has_missing_data:
            obs_np = np.asarray(obs_values)
            nan_per_row = np.isnan(obs_np).any(axis=1)
            all_nan_per_row = np.isnan(obs_np).all(axis=1)
            has_partial = (nan_per_row & ~all_nan_per_row).any()
            has_entire_row_missing = all_nan_per_row.any()
        elif not has_no_obs and self.unroll_missing and is_dirac:
            # unroll_missing=True with DiracIdentity: use scan even without NaN.
            obs_np = np.asarray(obs_values)
            has_partial = False
            has_entire_row_missing = False
        else:
            obs_np = None
            has_partial = False
            has_entire_row_missing = False

        if has_partial or (
            self.unroll_missing and (has_entire_row_missing or is_dirac)
        ):
            # Two-mask scan path: step through all T time steps.
            # obs_mask[t, j] = True  → score dim j at time t
            # latent_mask[t, i] = True → sample dim i at time t
            use_two_mask_scan = True
            if obs_np is None:
                obs_np = np.asarray(obs_values)
            obs_mask_np = ~np.isnan(obs_np)
            safe_obs_np = np.where(obs_mask_np, obs_np, 0.0)
            obs_mask_jax = jnp.array(obs_mask_np)
            safe_obs_jax = jnp.array(safe_obs_np)

            latent_mask_np: np.ndarray
            if isinstance(dynamics, WithPartialMissingnessSupport):
                latent_mask_np = np.asarray(dynamics.compute_latent_mask(obs_np))
            elif is_dirac:
                latent_mask_np = ~obs_mask_np
            else:
                latent_mask_np = np.ones((T, dynamics.state_dim), dtype=bool)
            latent_mask_jax = jnp.array(latent_mask_np)
        elif has_entire_row_missing:
            # Existing path: filter out entirely-missing rows.
            # state_evolution handles non-unit dt from skipped rows.
            assert obs_np is not None
            observed_mask = ~all_nan_per_row
            obs_values = jnp.array(obs_np[observed_mask])
            obs_times = jnp.array(np.asarray(obs_times)[observed_mask])
            if ctrl_values is not None:
                ctrl_values = jnp.array(np.asarray(ctrl_values)[observed_mask])
            T = len(obs_times)
            if T < 1:
                raise ValueError(
                    "obs_times must contain at least one timepoint after "
                    "removing missing data"
                )

        if use_two_mask_scan:
            # ------------------------------------------------------------------
            # Two-mask scan: step through every time step.
            # Sub-path A (DiracIdentity): score transition per dim via factor;
            #   pin carry to observed values; sample latent dims only.
            # Sub-path B (non-Dirac): sample all alive dims; score obs via
            #   masked_log_prob.
            # ------------------------------------------------------------------
            assert obs_mask_jax is not None
            assert safe_obs_jax is not None
            assert latent_mask_jax is not None
            u_0 = _get_val_or_None(ctrl_values, 0)
            obs_mask_0 = obs_mask_jax[0]
            safe_obs_0 = safe_obs_jax[0]
            latent_mask_0 = latent_mask_jax[0]

            # Whether any latent (unobserved) dims exist across all time steps.
            # False when obs_mask = all_True everywhere (fully-observed Dirac case).
            # Checked at Python time so we can select a sample-free step function.
            has_any_latent = bool(latent_mask_np.any()) if is_dirac else True

            if is_dirac:
                if has_any_latent:
                    # Some unobserved dims: sample all, then correction factor
                    # for observed dims (same approach as _step_masked).
                    x_0_full = numpyro.sample("x_0", dynamics.initial_condition)
                    per_dim_lp_0_obs = _per_dim_log_prob(
                        dynamics.initial_condition, safe_obs_0
                    )
                    per_dim_lp_0_samp = _per_dim_log_prob(
                        dynamics.initial_condition, x_0_full
                    )
                    numpyro.factor(
                        "x_0_obs_corr",
                        jnp.sum(
                            jnp.where(
                                obs_mask_0,
                                per_dim_lp_0_obs - per_dim_lp_0_samp,
                                0.0,
                            )
                        ),
                    )
                    x_prev: State = jnp.where(obs_mask_0, safe_obs_0, x_0_full)
                else:
                    # Fully observed: only factor log prob, carry = obs
                    per_dim_lp_0 = _per_dim_log_prob(
                        dynamics.initial_condition, safe_obs_0
                    )
                    numpyro.factor(
                        "x_0_obs_lp",
                        jnp.sum(jnp.where(obs_mask_0, per_dim_lp_0, 0.0)),
                    )
                    x_prev = safe_obs_0
            else:
                with numpyro.handlers.mask(mask=latent_mask_0):
                    x_0_full = numpyro.sample("x_0", dynamics.initial_condition)
                x_prev = jnp.where(latent_mask_0, x_0_full, jnp.nan)
                obs_model = dynamics.observation_model
                assert isinstance(obs_model, ObservationModel)
                obs_lp_0 = obs_model.masked_log_prob(
                    y=safe_obs_0, obs_mask=obs_mask_0, x=x_prev, u=u_0, t=obs_times[0]
                )
                numpyro.factor("y_0_lp", obs_lp_0)

            y_0 = jnp.where(obs_mask_0, safe_obs_0, jnp.nan)

            if is_dirac and has_any_latent:

                def _step_masked(x_prev, t_idx):
                    t_now = obs_times[t_idx]
                    t_next = obs_times[t_idx + 1]
                    u_now = _get_val_or_None(ctrl_values, t_idx)

                    obs_mask_t = obs_mask_jax[t_idx + 1]
                    safe_obs_t = safe_obs_jax[t_idx + 1]

                    trans_dist = dynamics.state_evolution(
                        x=x_prev, u=u_now, t_now=t_now, t_next=t_next
                    )
                    # Sample all dims jointly (needed to track latent state).
                    x_t_full = numpyro.sample(f"x_{t_idx + 1}", trans_dist)

                    # Correction factor: for observed dims, swap the model's
                    # log_prob of the sample for the log_prob of the observation.
                    # This leaves the latent dims' log_probs unchanged.
                    per_dim_lp_obs = _per_dim_log_prob(trans_dist, safe_obs_t)
                    per_dim_lp_samp = _per_dim_log_prob(trans_dist, x_t_full)
                    numpyro.factor(
                        f"x_{t_idx + 1}_obs_corr",
                        jnp.sum(
                            jnp.where(obs_mask_t, per_dim_lp_obs - per_dim_lp_samp, 0.0)
                        ),
                    )

                    # Carry: pin observed dims to data, keep sampled latent dims
                    x_t = jnp.where(obs_mask_t, safe_obs_t, x_t_full)
                    y_t = jnp.where(obs_mask_t, safe_obs_t, jnp.nan)
                    return x_t, (x_t, y_t)
            elif is_dirac:
                # Fully observed: factor transition log probs, carry = obs (no sample sites)
                def _step_masked(x_prev, t_idx):
                    t_now = obs_times[t_idx]
                    t_next = obs_times[t_idx + 1]
                    u_now = _get_val_or_None(ctrl_values, t_idx)

                    obs_mask_t = obs_mask_jax[t_idx + 1]
                    safe_obs_t = safe_obs_jax[t_idx + 1]

                    trans_dist = dynamics.state_evolution(
                        x=x_prev, u=u_now, t_now=t_now, t_next=t_next
                    )
                    per_dim_lp = _per_dim_log_prob(trans_dist, safe_obs_t)
                    numpyro.factor(
                        f"x_{t_idx + 1}_obs_lp",
                        jnp.sum(jnp.where(obs_mask_t, per_dim_lp, 0.0)),
                    )
                    y_t = jnp.where(obs_mask_t, safe_obs_t, jnp.nan)
                    return safe_obs_t, (safe_obs_t, y_t)
            else:

                def _step_masked(x_prev, t_idx):
                    t_now = obs_times[t_idx]
                    t_next = obs_times[t_idx + 1]
                    u_now = _get_val_or_None(ctrl_values, t_idx)
                    u_next = _get_val_or_None(ctrl_values, t_idx + 1)

                    latent_mask_t = latent_mask_jax[t_idx + 1]
                    with numpyro.handlers.mask(mask=latent_mask_t):
                        x_t_full = numpyro.sample(
                            f"x_{t_idx + 1}",
                            dynamics.state_evolution(
                                x=x_prev, u=u_now, t_now=t_now, t_next=t_next
                            ),
                        )
                    x_t = jnp.where(latent_mask_t, x_t_full, jnp.nan)

                    obs_mask_t = obs_mask_jax[t_idx + 1]
                    safe_obs_t = safe_obs_jax[t_idx + 1]
                    _obs_model = dynamics.observation_model
                    assert isinstance(_obs_model, ObservationModel)
                    obs_lp = _obs_model.masked_log_prob(
                        y=safe_obs_t, obs_mask=obs_mask_t, x=x_t, u=u_next, t=t_next
                    )
                    numpyro.factor(f"y_{t_idx + 1}_lp", obs_lp)

                    y_t = jnp.where(obs_mask_t, safe_obs_t, jnp.nan)
                    return x_t, (x_t, y_t)

            _, scan_outputs = nscan(_step_masked, x_prev, jnp.arange(T - 1))
            scan_states, scan_observations = scan_outputs

            x_0_expanded = jnp.expand_dims(x_prev, axis=0)
            y_0_expanded = jnp.expand_dims(y_0, axis=0)
            states = jnp.concatenate([x_0_expanded, scan_states], axis=0)
            observations = jnp.concatenate([y_0_expanded, scan_observations], axis=0)

            return {"times": obs_times, "states": states, "observations": observations}

        if (
            isinstance(dynamics.observation_model, DiracIdentityObservation)
            and not has_no_obs
        ):
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
        y_0 = numpyro.sample(  # type: ignore[assignment]
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
