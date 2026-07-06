"""Auto-routing simulator backend."""

from jax import Array

from dynestyx.inference.configs.simulator import (
    ODESimulatorConfig,
    SDESimulatorConfig,
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
        - `ode_simulator_config` and `sde_simulator_config` can be supplied to
          provide structured per-backend settings while still relying on
          auto-routing.
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
        *args,
        ode_simulator_config: ODESimulatorConfig | None = None,
        sde_simulator_config: SDESimulatorConfig | None = None,
        **kwargs,
    ):
        self.args = args
        self.kwargs = kwargs
        self.ode_simulator_config = ode_simulator_config
        self.sde_simulator_config = sde_simulator_config

        self.simulator: BaseSimulator | None = None

    def _ensure_simulator(self, dynamics: DynamicalModel) -> BaseSimulator:
        """Instantiate and cache the concrete simulator for ``dynamics``."""
        if self.simulator is not None:
            return self.simulator

        if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
            if self.sde_simulator_config is not None:
                self.simulator = SDESimulator(
                    *self.args,
                    simulator_config=self.sde_simulator_config,
                    **self.kwargs,
                )
            else:
                self.simulator = SDESimulator(*self.args, **self.kwargs)
        elif isinstance(
            dynamics.state_evolution, DeterministicContinuousTimeStateEvolution
        ):
            if self.ode_simulator_config is not None:
                self.simulator = ODESimulator(
                    *self.args,
                    simulator_config=self.ode_simulator_config,
                    **self.kwargs,
                )
            else:
                self.simulator = ODESimulator(*self.args, **self.kwargs)
        else:
            # Non-continuous models are discrete-time. This includes structured
            # DiscreteTimeStateEvolution instances and plain transition callables.
            self.simulator = DiscreteTimeSimulator(*self.args, **self.kwargs)

        return self.simulator

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
        simulator = self._ensure_simulator(dynamics)
        return simulator._simulate(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            **kwargs,
        )

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
        """Auto-route to the appropriate pure-JAX simulator backend."""
        simulator = self._ensure_simulator(dynamics)
        return simulator.simulate(
            dynamics,
            rng_key=rng_key,
            obs_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            predict_times=predict_times,
            **kwargs,
        )
