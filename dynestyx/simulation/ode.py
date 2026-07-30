"""ODE forward-simulation backend."""

import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.inference.configs.simulator import ODESimulatorConfig
from dynestyx.models import DynamicalModel
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.utils import _sample_initial_states, _tile_times
from dynestyx.solvers import solve_ode_state_path
from dynestyx.types import SimulatedResult
from dynestyx.utils import _build_control_path_eval


class ODESimulator(BaseSimulator):
    r"""Generate trajectories from deterministic continuous-time dynamics.

    For an initial-condition distribution \(p_0\), drift function \(f\), and
    observation model \(p(y\mid x,u,t)\), `ODESimulator` draws
    `n_simulations` independent initial states and computes

    \[
    x_0^{(m)} \sim p_0(x_0), \qquad
    \frac{\mathrm{d}x^{(m)}(t)}{\mathrm{d}t}
      = f\!\left(x^{(m)}(t),u(t),t\right), \qquad
    y_k^{(m)} \sim p\!\left(y_k\mid x^{(m)}(t_k),u(t_k),t_k\right).
    \]

    The ODE solution is evaluated at every value in `predict_times`. Conditional
    on the initial state and controls, the state path is deterministic; the
    initial-condition and observation distributions may still make the complete
    simulation stochastic. See
    [ContinuousTimeStateEvolution][dynestyx.models.core.ContinuousTimeStateEvolution]
    for how an ODE is represented in a `DynamicalModel` by specifying its drift
    without a diffusion.

    Use `ODESimulator` as a context manager around a model containing
    `dsx.sample(name, dynamics, predict_times=...)`. The active NumPyro seed
    supplies randomness, and the computed arrays are then attached to the trace
    as deterministic sites. Pass an
    [ODESimulatorConfig][dynestyx.inference.configs.simulator.ODESimulatorConfig]
    to choose the Diffrax solver, step-size controller, adjoint, step size, and
    step limit. Use [dsx.simulate][dynestyx.api.simulate] for standalone
    pure-JAX generation without a NumPyro trace.

    Examples:
        Prior-predictive ODE trajectories:

        >>> def model(predict_times=None):
        ...     dynamics = DynamicalModel(
        ...         initial_condition=initial_dist,
        ...         state_evolution=ContinuousTimeStateEvolution(
        ...             drift=lambda x, u, t: -rate * x,
        ...         ),
        ...         observation_model=observation,
        ...     )
        ...     dsx.sample("f", dynamics, predict_times=predict_times)
        >>> config = ODESimulatorConfig(dt0=1e-2)
        >>> with ODESimulator(config, n_simulations=3):
        ...     predictive = Predictive(
        ...         model, num_samples=10, exclude_deterministic=False
        ...     )
        ...     draws = predictive(
        ...         jr.PRNGKey(0), predict_times=jnp.linspace(0.0, 5.0, 51)
        ...     )
        >>> draws["f_states"].shape
        (10, 3, 51, state_dim)

        Standalone pure-JAX simulation uses the same ODE solver:

        >>> result = dsx.simulate(
        ...     dynamics,
        ...     rng_key=jr.PRNGKey(0),
        ...     predict_times=times,
        ...     n_simulations=3,
        ...     simulator_config=ODESimulatorConfig(dt0=1e-2),
        ... )

    What this does
    --------------
    Each initial-condition draw is integrated independently with Diffrax. The
    integration starts at `dynamics.t0` when it is defined and otherwise at the
    first prediction time. The solved state is saved only at `predict_times`,
    after which the observation model is sampled independently at those states.

    If controls are supplied, they form a right-continuous rectilinear path:
    the control at a knot `ctrl_times[k]` is `ctrl_values[k]`, and that value is
    held until the next knot.

    This handler is generation-only and does not condition on `obs_times` or
    `obs_values`. Use
    [LatentPathBuilder][dynestyx.inference.latent.builder.LatentPathBuilder]
    for explicit latent-path inference, or use
    [Filter][dynestyx.inference.filters.Filter] or
    [Smoother][dynestyx.inference.smoothers.Smoother] for marginalized
    inference. Placing this simulator outside a compatible continuous-time
    `Filter` or `Smoother` draws posterior rollouts at `predict_times`.

    Configuration and defaults
    --------------------------
    ODEs are solved using Diffrax, and settings are controlled by
    [ODESimulatorConfig][dynestyx.inference.configs.simulator.ODESimulatorConfig].
    Its default settings are `diffrax.Tsit5()`,
    `diffrax.ConstantStepSize()`, `diffrax.RecursiveCheckpointAdjoint()`,
    `dt0=1e-3`, and `max_steps=100_000`. Pass different settings when the model
    requires them. If `simulator_config=None`, a default
    `ODESimulatorConfig()` is created.

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
    [ODESimulator().simulate][dynestyx.simulation.ode.ODESimulator.simulate]
    return `SimulatedResult` without adding NumPyro sites.

    Notes:
        - Use `Simulator` instead when automatic selection among discrete, ODE,
          and SDE backends is desirable.
        - `ODESimulator().simulate(...)` consumes an already allocated simulation
          key. The public [dsx.simulate][dynestyx.api.simulate] function splits
          its root key before dispatch.

    Attributes:
        simulator_config: ODE solver and integration settings. Defaults to
            `ODESimulatorConfig()`.
        n_simulations: Number of independent initial states and trajectories
            drawn per model execution. Defaults to one and must be greater than
            or equal to one.
        diffeqsolve_settings: Normalized settings passed to Diffrax.
    """

    def __init__(
        self,
        simulator_config: ODESimulatorConfig | None = None,
        *,
        n_simulations: int = 1,
    ) -> None:
        """Configure ODE integration.

        Args:
            simulator_config: Structured simulator settings. Defaults to
                `ODESimulatorConfig()` when omitted.
            n_simulations: Number of independent trajectories to simulate. State
                and observation paths have shape `(n_simulations, T, ...)`. Must
                be greater than or equal to one.
        """
        super().__init__(n_simulations=n_simulations)
        if simulator_config is None:
            simulator_config = ODESimulatorConfig()

        self.simulator_config = simulator_config
        self.diffeqsolve_settings = simulator_config.diffeqsolve_settings

    def _simulate_forward_from_initial_state(
        self,
        dynamics: DynamicalModel,
        *,
        initial_state: Real[Array, "n_simulations state_dim"]
        | Real[Array, " n_simulations"],
        rng_key: PRNGKeyArray,
        times: Real[Array, " time"],
        ctrl_times: Real[Array, " ctrl_time"] | None = None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None = None,
    ) -> SimulatedResult:
        """Run pure forward simulation for a deterministic continuous-time model."""
        n_sim = initial_state.shape[0]

        control_path_eval = _build_control_path_eval(ctrl_times, ctrl_values, times)

        t0 = dynamics.t0 if dynamics.t0 is not None else times[0]
        obs_keys = jr.split(rng_key, n_sim)

        def _sim_one_trajectory(
            x0: Real[Array, " state_dim"] | Real[Array, ""],
            *,
            obs_key: PRNGKeyArray,
        ):
            states = solve_ode_state_path(
                dynamics,
                t0=t0,
                initial_state=x0,
                path_times=times,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                diffeqsolve_settings=self.diffeqsolve_settings,
            )
            observations = self._emit_observations(
                "",
                dynamics,
                states,
                times,
                None,
                control_path_eval,
                key=obs_key,
            )
            return states, observations

        states, observations = jax.vmap(_sim_one_trajectory)(
            initial_state, obs_key=obs_keys
        )
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
        rng_key: PRNGKeyArray,
        ctrl_times: Real[Array, " ctrl_time"] | None = None,
        ctrl_values: Real[Array, "ctrl_time control_dim"]
        | Real[Array, " ctrl_time"]
        | None = None,
        predict_times: Real[Array, " predict_time"] | None = None,
        **kwargs,
    ) -> SimulatedResult:
        """Run pure-JAX forward simulation for deterministic continuous-time models.

        Unlike [dsx.simulate][dynestyx.api.simulate], `rng_key` is consumed
        directly as an already-allocated simulation key and is not pre-split.
        Therefore, `dsx.simulate(..., rng_key=root_key)` is equivalent to
        `ODESimulator().simulate(..., rng_key=jax.random.split(root_key)[1])`.
        """
        if predict_times is None:
            raise ValueError("predict_times must be provided")

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
            times=predict_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
        )
