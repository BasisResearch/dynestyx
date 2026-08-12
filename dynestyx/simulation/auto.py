"""Auto-routing simulator handler."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jaxtyping import Array, PRNGKeyArray, PyTree, Real

from dynestyx.inference.configs.filter import BaseFilterConfig
from dynestyx.inference.configs.simulator import (
    ODESimulatorConfig,
    SDESimulatorConfig,
    SimulatorConfig,
)
from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.simulation.base import BaseSimulator
from dynestyx.simulation.discrete import DiscreteTimeSimulator
from dynestyx.simulation.ode import ODESimulator
from dynestyx.simulation.sde import SDESimulator
from dynestyx.types import SimulatedResult

if TYPE_CHECKING:
    # Deferred: dynestyx.control imports from dynestyx.simulation.base, so a
    # top-level import here would be circular. Only needed for type-checking.
    from dynestyx.control.discrete_controller_simulators import PolicyCallable


class Simulator(BaseSimulator):
    r"""Generate trajectories using the simulator appropriate for the model.

    `Simulator` is the auto-routing simulator handler. Given prediction times
    \(t_{0:T-1}\), it draws `n_simulations` independent trajectories from the
    dynamical model:

    \[
    x_0^{(m)} \sim p_0(x_0), \qquad
    x_{1:T-1}^{(m)}
      \sim p(x_{1:T-1}\mid x_0^{(m)},u_{1:T-1},t_{1:T-1}),
    \qquad
    y_k^{(m)} \sim p(y_k\mid x_k^{(m)}, u_k, t_k).
    \]

    For continuous-time models, the middle term denotes a numerical ODE or SDE
    solve evaluated at `predict_times`. For discrete-time models, it denotes
    successive draws from the transition distribution between adjacent
    prediction times.

    Use `Simulator` as a context manager around model execution when the model
    contains `dsx.sample(name, dynamics, predict_times=...)`. The active NumPyro
    seed supplies randomness; the rollout itself is computed with pure JAX and
    its realized arrays are then attached to the NumPyro trace. For
    continuous-time models, pass an
    [ODESimulatorConfig][dynestyx.inference.configs.simulator.ODESimulatorConfig]
    or
    [SDESimulatorConfig][dynestyx.inference.configs.simulator.SDESimulatorConfig]
    as `simulator_config` to control how the differential equation is solved.
    Use [dsx.simulate][dynestyx.api.simulate] instead when no NumPyro trace is
    needed and an explicit `rng_key` is more convenient.

    Examples:
        Prior-predictive trajectories with automatic backend selection:

        >>> def model(predict_times=None):
        ...     dynamics = DynamicalModel(...)
        ...     dsx.sample("f", dynamics, predict_times=predict_times)
        >>> with Simulator(n_simulations=4):
        ...     predictive = Predictive(
        ...         model, num_samples=20, exclude_deterministic=False
        ...     )
        ...     draws = predictive(jr.PRNGKey(0), predict_times=times)
        >>> draws["f_states"].shape
        (20, 4, T, state_dim)

        Pure-JAX forward simulation returns the same trajectory fields without
        adding NumPyro sites:

        >>> result = dsx.simulate(
        ...     dynamics,
        ...     rng_key=jr.PRNGKey(0),
        ...     predict_times=times,
        ...     n_simulations=4,
        ... )

        For posterior state rollouts, place the simulator outside the inference
        handler and retain deterministic sites from `Predictive`:

        >>> with Simulator(n_simulations=100):
        ...     with Filter():
        ...         predictive = Predictive(
        ...             model,
        ...             posterior_samples=posterior_samples,
        ...             exclude_deterministic=False,
        ...         )
        ...         forecast = predictive(
        ...             jr.PRNGKey(1),
        ...             obs_times=obs_times,
        ...             obs_values=obs_values,
        ...             predict_times=forecast_times,
        ...         )

    What this does
    --------------
    If `control_policy` is given, the backend is always
    [DiscreteControlLoopSimulator][dynestyx.control.discrete_controller_simulators.DiscreteControlLoopSimulator],
    regardless of `dynamics.state_evolution`. Otherwise the concrete backend is
    selected from `dynamics.state_evolution`:

    - `StochasticContinuousTimeStateEvolution` uses
      [SDESimulator][dynestyx.simulation.sde.SDESimulator].
    - `DeterministicContinuousTimeStateEvolution` uses
      [ODESimulator][dynestyx.simulation.ode.ODESimulator].
    - Discrete-time state evolution uses
      [DiscreteTimeSimulator][dynestyx.simulation.discrete.DiscreteTimeSimulator].

    A simulator is generation-only: raw simulator calls accept
    `predict_times`, not `obs_times` or `obs_values`. For observation-conditioned
    inference, use
    [LatentPathBuilder][dynestyx.inference.latent.builder.LatentPathBuilder],
    [Filter][dynestyx.inference.filters.Filter], or
    [Smoother][dynestyx.inference.smoothers.Smoother]. A simulator may wrap a
    `Filter` or `Smoother` to draw posterior rollouts at `predict_times`; in
    that composition, the inference handler consumes the observations and
    passes posterior state distributions outward to the simulator.

    Configurations and defaults
    ---------------------------
    `simulator_config` accepts a
    [SimulatorConfig][dynestyx.inference.configs.simulator.SimulatorConfig] and
    forwards it to the selected continuous-time backend:

    - stochastic continuous-time models accept `SDESimulatorConfig`;
    - deterministic continuous-time models accept `ODESimulatorConfig`;
    - discrete-time models do not accept a simulator config.

    If `simulator_config=None`, discrete transitions are sampled directly,
    deterministic continuous-time models use the `ODESimulatorConfig` defaults
    (`diffrax.Tsit5()` with fixed `dt0=1e-3`), and stochastic continuous-time
    models use the `SDESimulatorConfig` defaults (the fixed-step `"em_scan"`
    Euler--Maruyama backend with `dt0=1e-4`). `n_simulations` defaults to one
    and must be at least one. The simulation dimension is retained even when it
    has length one.

    NumPyro trace
    -------------
    For a raw rollout from `dsx.sample("f", ...)`, the following
    `numpyro.deterministic` sites are added:

    - `"f_x_0"`: initial states, shape
      `(*plate_shape, n_simulations, state_dim)`;
    - `"f_times"`: prediction times, shape
      `(*plate_shape, n_simulations, T)`;
    - `"f_states"`: latent states, shape
      `(*plate_shape, n_simulations, T, state_dim)`;
    - `"f_observations"`: sampled observations, shape
      `(*plate_shape, n_simulations, T, observation_dim)`.

    Here `"f"` is replaced by the `name` passed to `dsx.sample`, `T` is the
    length of the prediction grid, and `plate_shape` is absent outside a
    `dsx.plate`. Under `Predictive(..., num_samples=N)`, NumPyro prepends an
    `N` axis to every shape above. Because these sites are deterministic, pass
    `exclude_deterministic=False` to `Predictive` (or request the site names
    explicitly) to include them in its returned dictionary.

    When the simulator wraps a `Filter` or `Smoother`, the inner handler records
    its own configured sites and the simulator's aggregate rollout sites are
    instead `"f_predicted_times"`, `"f_predicted_states"`, and
    `"f_predicted_observations"`, with the corresponding time, state, and
    observation shapes above. Each nonempty prediction segment also records
    the state from which that segment starts, with shape
    `(n_simulations, state_dim)`: `"f_0_x_0"` for a segment before the first
    posterior time, and `"f_{j+1}_x_0"` for a segment initialized from the
    posterior at inference-time index `j`. Only segments containing at least
    one requested prediction time are recorded. Inside `dsx.plate`, the segment
    name also identifies the plate member, for example `"f_p0_1_x_0"`.

    If `predict_times` is omitted, the simulator performs no rollout and adds
    no simulator sites. Direct calls to
    [Simulator().simulate][dynestyx.simulation.auto.Simulator.simulate] and
    [dsx.simulate][dynestyx.api.simulate] return a `SimulatedResult` and do not
    add NumPyro sites.

    Notes:
        - Controls are passed through to the selected backend. See the concrete
          simulator for its control interpolation or alignment rules.
        - A `Simulator` instance selects and caches its concrete backend on the
          first dynamics object it handles. Use separate instances for models
          requiring different backend types.
        - `Simulator().simulate(...)` consumes an already allocated simulation
          key. The public [dsx.simulate][dynestyx.api.simulate] function splits
          its root key before dispatch.

    Attributes:
        simulator_config: Optional ODE or SDE solver configuration. Its type
            must match the model selected at first use.
        n_simulations: Number of independent trajectories drawn per model
            execution. Defaults to one and must be greater than or equal to one.
        control_policy: Optional control policy (see
            `dynestyx.control.discrete_controller_simulators.PolicyCallable`).
            When given, routing ignores `dynamics.state_evolution`'s type
            entirely and always uses
            [DiscreteControlLoopSimulator][dynestyx.control.discrete_controller_simulators.DiscreteControlLoopSimulator]
            instead -- a policy is an orthogonal choice from the dynamics
            themselves, not something inferable from `dynamics`.
        filter_config: Filter configuration forwarded to
            `DiscreteControlLoopSimulator` when `control_policy` is given;
            ignored otherwise.
        simulator: Concrete auto-selected simulator cached on first use.
    """

    def __init__(
        self,
        simulator_config: SimulatorConfig | None = None,
        *,
        n_simulations: int = 1,
        control_policy: PolicyCallable | None = None,
        filter_config: BaseFilterConfig | None = None,
    ) -> None:
        super().__init__(n_simulations=n_simulations)
        self.simulator_config = simulator_config
        self.control_policy = control_policy
        self.filter_config = filter_config
        self.simulator: BaseSimulator | None = None

    def _ensure_simulator(self, dynamics: DynamicalModel) -> BaseSimulator:
        """Instantiate and cache the concrete simulator for ``dynamics``."""
        if self.simulator is not None:
            return self.simulator

        if self.control_policy is not None:
            from dynestyx.control.discrete_controller_simulators import (
                DiscreteControlLoopSimulator,
            )

            if self.simulator_config is not None:
                raise ValueError(
                    "Received a SimulatorConfig together with control_policy. "
                    "DiscreteControlLoopSimulator does not accept a simulator_config."
                )
            self.simulator = DiscreteControlLoopSimulator(
                control_policy=self.control_policy,
                filter_config=self.filter_config,
                n_simulations=self.n_simulations,
            )
        elif isinstance(
            dynamics.state_evolution, StochasticContinuousTimeStateEvolution
        ):
            if isinstance(self.simulator_config, ODESimulatorConfig):
                raise ValueError(
                    "Received an ODESimulatorConfig for stochastic continuous-time "
                    "dynamics. Pass an SDESimulatorConfig instead."
                )
            self.simulator = SDESimulator(
                simulator_config=self.simulator_config,
                n_simulations=self.n_simulations,
            )
        elif isinstance(
            dynamics.state_evolution, DeterministicContinuousTimeStateEvolution
        ):
            if isinstance(self.simulator_config, SDESimulatorConfig):
                raise ValueError(
                    "Received an SDESimulatorConfig for deterministic continuous-time "
                    "dynamics. Pass an ODESimulatorConfig instead."
                )
            self.simulator = ODESimulator(
                simulator_config=self.simulator_config,
                n_simulations=self.n_simulations,
            )
        else:
            if self.simulator_config is not None:
                raise ValueError(
                    "Received a continuous-time SimulatorConfig for discrete-time "
                    "dynamics. Use direct DiscreteTimeSimulator settings instead."
                )
            self.simulator = DiscreteTimeSimulator(n_simulations=self.n_simulations)

        return self.simulator

    def _validate_plate_support(self) -> None:
        """Reject plated closed-loop control until aggregation is supported."""
        if self.control_policy is not None:
            raise NotImplementedError(
                "Simulator(control_policy=...) does not yet support dsx.plate. "
                "Run one controlled model at a time."
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
        initial_policy_state: PyTree | None = None,
        **kwargs: Any,
    ) -> SimulatedResult:
        """Auto-route to the appropriate pure-JAX simulator backend.

        Unlike [dsx.simulate][dynestyx.api.simulate], `rng_key` is consumed
        directly as an already-allocated simulation key and is not pre-split.
        Therefore, `dsx.simulate(..., rng_key=root_key)` is equivalent to
        `Simulator().simulate(..., rng_key=jax.random.split(root_key)[1])`.

        `initial_policy_state` is forwarded to the controlled simulator when
        `control_policy` was supplied to this `Simulator`; it is ignored for
        ordinary open-loop simulation.
        """
        simulator = self._ensure_simulator(dynamics)
        if self.control_policy is not None:
            kwargs["initial_policy_state"] = initial_policy_state
        return simulator.simulate(
            dynamics,
            rng_key=rng_key,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            **kwargs,
        )
