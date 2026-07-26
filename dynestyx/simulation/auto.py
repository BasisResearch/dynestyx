"""Auto-routing simulator handler."""

from jax import Array

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


class Simulator(BaseSimulator):
    """Auto-selecting simulator wrapper.

    Chooses a concrete simulator based on the structure of
    ``dynamics.state_evolution``:

    - stochastic continuous-time -> ``SDESimulator``
    - deterministic continuous-time -> ``ODESimulator``
    - otherwise -> ``DiscreteTimeSimulator``
    """

    def __init__(
        self,
        simulator_config: SimulatorConfig | None = None,
        *,
        n_simulations: int = 1,
    ):
        super().__init__(n_simulations=n_simulations)
        self.simulator_config = simulator_config
        self.simulator: BaseSimulator | None = None

    def _ensure_simulator(self, dynamics: DynamicalModel) -> BaseSimulator:
        """Instantiate and cache the concrete simulator for ``dynamics``."""
        if self.simulator is not None:
            return self.simulator

        if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
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
        """Auto-route to the appropriate pure-JAX simulator backend.

        Unlike :func:`dynestyx.simulate`, ``rng_key`` is consumed directly as an
        already-allocated simulation key and is not pre-split. Therefore,
        ``dynestyx.simulate(..., rng_key=root_key)`` is equivalent to
        ``Simulator.simulate(..., rng_key=jax.random.split(root_key)[1])``.
        """
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
