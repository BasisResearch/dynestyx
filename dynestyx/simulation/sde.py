"""SDE simulator backend."""

import jax
import jax.random as jr
from jax import Array

from dynestyx.inference.configs.simulator import SDESimulatorConfig
from dynestyx.models import DynamicalModel, StochasticContinuousTimeStateEvolution
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import _sample_initial_states, _tile_times
from dynestyx.solvers import solve_sde_state_path
from dynestyx.types import SimulatedResult
from dynestyx.utils import _build_control_path_eval


class SDESimulator(BaseSimulator):
    r"""Generate trajectories from stochastic continuous-time dynamics.

    For an initial-condition distribution \(p(x_0)\), drift \(f\), diffusion
    coefficient \(g\), and observation model \(p(y\mid x,u,t)\),
    `SDESimulator` draws `n_simulations` independent paths satisfying

    \[
    \begin{aligned}
    x_0^{(m)} &\sim p(x_0), \\
    dx_t^{(m)}
      &= f(x_t^{(m)},u_t,t)\,dt
       + g(x_t^{(m)},u_t,t)\,dW_t^{(m)}, \\
    y_k^{(m)}
      &\sim p(y_k\mid x_k^{(m)},u_k,t_k).
    \end{aligned}
    \]

    The numerical SDE solution is evaluated at every value in
    `predict_times`. Initial states, Brownian paths, and observations are drawn
    independently across the simulation dimension. See
    [ContinuousTimeStateEvolution][dynestyx.models.core.ContinuousTimeStateEvolution]
    for how an SDE is represented in a `DynamicalModel`.

    Use `SDESimulator` as a context manager around a model containing
    `dsx.sample(name, dynamics, predict_times=...)`. The active NumPyro seed
    supplies randomness, and the realized paths are then attached to the trace
    as deterministic sites. Pass an
    [SDESimulatorConfig][dynestyx.inference.configs.simulator.SDESimulatorConfig]
    to choose the SDE backend, solver, step-size controller, adjoint, step size,
    and Brownian-tree tolerance. Use
    [dsx.simulate][dynestyx.api.simulate] for standalone pure-JAX generation
    without a NumPyro trace.

    Examples:
        Fast fixed-step Euler--Maruyama simulation:

        >>> def model(predict_times=None):
        ...     dynamics = DynamicalModel(
        ...         initial_condition=initial_dist,
        ...         state_evolution=ContinuousTimeStateEvolution(
        ...             drift=drift,
        ...             diffusion=diffusion,
        ...         ),
        ...         observation_model=observation,
        ...     )
        ...     dsx.sample("f", dynamics, predict_times=predict_times)
        >>> config = SDESimulatorConfig(source="em_scan", dt0=1e-3)
        >>> with SDESimulator(config, n_simulations=4):
        ...     predictive = Predictive(
        ...         model, num_samples=10, exclude_deterministic=False
        ...     )
        ...     draws = predictive(
        ...         jr.PRNGKey(0), predict_times=jnp.linspace(0.0, 5.0, 51)
        ...     )
        >>> draws["f_states"].shape
        (10, 4, 51, state_dim)

        Use the Diffrax backend when a different SDE solver or step-size
        controller is required:

        >>> config = SDESimulatorConfig(
        ...     source="diffrax",
        ...     solver=diffrax.EulerHeun(),
        ...     dt0=1e-3,
        ... )

    What this does
    --------------
    Each initial-condition draw is integrated with an independent Brownian
    path. Integration starts at `dynamics.t0` when it is defined and otherwise
    at the first prediction time. After the state paths are solved, the
    observation model is sampled independently at the requested times.

    If controls are supplied, they form a right-continuous rectilinear path:
    the control at a knot `ctrl_times[k]` is `ctrl_values[k]`, and that value is
    held until the next knot.

    This handler is generation-only and does not condition on `obs_times` or
    `obs_values`. Native SDE latent-path inference should go through a
    [Discretizer][dynestyx.discretizers.Discretizer] and
    [LatentPathBuilder][dynestyx.inference.latent.builder.LatentPathBuilder];
    use
    [Filter][dynestyx.inference.filters.Filter] or
    [Smoother][dynestyx.inference.smoothers.Smoother] for marginalized
    inference. Placing this simulator outside a compatible continuous-time
    `Filter` or `Smoother` draws posterior rollouts at `predict_times`.

    Configurations and defaults
    ---------------------------
    [SDESimulatorConfig][dynestyx.inference.configs.simulator.SDESimulatorConfig]
    selects one of two backends:

    - `source="em_scan"` uses a fixed-step Euler--Maruyama `jax.lax.scan`.
      It is the default and uses `dt0=1e-4`.
    - `source="diffrax"` uses the configured Diffrax solver, step-size
      controller, adjoint, and virtual Brownian tree. Defaults include
      `diffrax.Heun()`, `diffrax.ConstantStepSize()`,
      `diffrax.RecursiveCheckpointAdjoint()`, and `dt0=1e-4`.
      `tol_vbt=None` resolves to `dt0 / 2`.

    Solver choice determines the stochastic integral represented by the
    numerical solution. In particular, the default Diffrax Heun solver
    converges to a Stratonovich solution, while Euler--Maruyama converges to an
    Itô solution. This distinction matters for state-dependent diffusion.

    If `simulator_config=None`, a default `SDESimulatorConfig()` is created.
    `n_simulations` defaults to one and must be at least one. The simulation
    dimension is retained even when it has length one.

    NumPyro trace
    -------------
    For a raw rollout from `dsx.sample("f", ...)`, the following
    `numpyro.deterministic` sites are added:

    - `"f_x_0"`: initial states, shape
      `(*plate_shape, n_simulations, state_dim)`;
    - `"f_times"`: prediction times, shape
      `(*plate_shape, n_simulations, T)`;
    - `"f_states"`: solved states, shape
      `(*plate_shape, n_simulations, T, state_dim)`;
    - `"f_observations"`: sampled observations, shape
      `(*plate_shape, n_simulations, T, observation_dim)`.

    Here `"f"` is replaced by the `name` passed to `dsx.sample`. Under
    `Predictive(..., num_samples=N)`, NumPyro prepends an `N` axis to each
    shape. Because these sites are deterministic, pass
    `exclude_deterministic=False` to `Predictive` (or request the site names
    explicitly) to include them in its returned dictionary.

    When this simulator wraps a `Filter` or `Smoother`, the inner handler
    records its own configured sites and the simulator's aggregate rollout
    sites are instead `"f_predicted_times"`, `"f_predicted_states"`, and
    `"f_predicted_observations"`, with the corresponding time, state, and
    observation shapes above. Each nonempty prediction segment also records
    the state from which that segment starts, with shape
    `(n_simulations, state_dim)`: `"f_0_x_0"` for a segment before the first
    posterior time, and `"f_{j+1}_x_0"` for a segment initialized from the
    posterior at inference-time index `j`. Only segments containing at least
    one requested prediction time are recorded. Inside `dsx.plate`, the segment
    name also identifies the plate member, for example `"f_p0_1_x_0"`.

    If `predict_times` is omitted, no simulator rollout or simulator trace
    sites are produced. Direct calls to
    [SDESimulator().simulate][dynestyx.simulation.sde.SDESimulator.simulate]
    return `SimulatedResult` without adding NumPyro sites.

    Notes:
        - Use `Simulator` instead when automatic selection among discrete, ODE,
          and SDE backends is desirable.
        - `SDESimulator().simulate(...)` consumes an already allocated simulation
          key. The public [dsx.simulate][dynestyx.api.simulate] function splits
          its root key before dispatch.

    Attributes:
        simulator_config: SDE backend and integration settings. Defaults to
            `SDESimulatorConfig()`.
        n_simulations: Number of independent initial states, Brownian paths,
            and trajectories drawn per model execution. Defaults to one and
            must be greater than or equal to one.
        source: Active SDE backend, either `"em_scan"` or `"diffrax"`.
        diffeqsolve_settings: Normalized Diffrax-style solver settings.
        tol_vbt: Resolved virtual-Brownian-tree tolerance for the Diffrax
            backend, or `None` for `"em_scan"`.
    """

    def __init__(
        self,
        simulator_config: SDESimulatorConfig | None = None,
        *,
        n_simulations: int = 1,
    ):
        """Configure SDE integration settings.

        Args:
            simulator_config: Structured simulator settings. Defaults to
                `SDESimulatorConfig()` when omitted.
            n_simulations: Number of independent trajectories to simulate. State
                and observation paths have shape `(n_simulations, T, ...)`. Must
                be greater than or equal to one.
        """
        super().__init__(n_simulations=n_simulations)
        if simulator_config is None:
            simulator_config = SDESimulatorConfig()

        self.simulator_config = simulator_config
        self.diffeqsolve_settings = simulator_config.diffeqsolve_settings
        self.source = simulator_config.source
        self.tol_vbt = simulator_config.resolved_tol_vbt

    def _simulate_forward_from_initial_state(
        self,
        dynamics: DynamicalModel,
        *,
        initial_state: Array,
        rng_key: Array,
        times: Array,
        ctrl_times=None,
        ctrl_values=None,
    ) -> SimulatedResult:
        """Run pure forward SDE simulation from provided initial states."""
        n_sim = initial_state.shape[0]

        control_path_eval = _build_control_path_eval(ctrl_times, ctrl_values, times)

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]
        sim_keys = jr.split(rng_key, n_sim)

        def _sim_one_trajectory(key: Array, x0: Array) -> tuple[Array, Array]:
            k_solve, k_obs = jr.split(key, 2)
            states = solve_sde_state_path(
                source=self.source,
                dynamics=dynamics,
                t0=t0,
                path_times=times,
                initial_state=x0,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                diffeqsolve_settings=self.diffeqsolve_settings,
                key=k_solve,
                tol_vbt=self.tol_vbt,
            )
            observations = self._emit_observations(
                "",
                dynamics,
                states,
                times,
                None,
                control_path_eval,
                key=k_obs,
            )
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(sim_keys, initial_state)
        return SimulatedResult(
            times=_tile_times(times, n_sim),
            x_0=initial_state,
            states=states,
            observations=observations,
        )

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        rng_key: Array,
        ctrl_times=None,
        ctrl_values=None,
        predict_times=None,
        **kwargs,
    ) -> SimulatedResult:
        """Run pure-JAX forward simulation for stochastic continuous-time models.

        Unlike [dsx.simulate][dynestyx.api.simulate], `rng_key` is consumed
        directly as an already-allocated simulation key and is not pre-split.
        Therefore, `dsx.simulate(..., rng_key=root_key)` is equivalent to
        `SDESimulator().simulate(..., rng_key=jax.random.split(root_key)[1])`.
        """
        if not isinstance(
            dynamics.state_evolution, StochasticContinuousTimeStateEvolution
        ):
            raise NotImplementedError(
                "SDESimulator only works with StochasticContinuousTimeStateEvolution, got "
                f"{type(dynamics.state_evolution)}"
            )

        times = predict_times
        if times is None:
            raise ValueError("predict_times must be provided for SDESimulator.")

        initial_key, rollout_key = jr.split(rng_key)
        initial_state = _sample_initial_states(
            dynamics.initial_condition,
            rng_key=initial_key,
            n_simulations=self.n_simulations,
        )
        return self._simulate_forward_from_initial_state(
            dynamics,
            initial_state=initial_state,
            rng_key=rollout_key,
            times=times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
