# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

from typing import TypeVar

import numpyro
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import NotHandled

from dynestyx.discretizers import euler_maruyama
from dynestyx.dynamical_models import (
    Context,
    ContinuousTimeStateEvolution,
    DynamicalModel,
    FunctionOfTime,
    State,
)

T = TypeVar("T")


@defop
def sample(
    name: str, dynamics: DynamicalModel, context: Context | None = None
) -> FunctionOfTime:
    raise NotHandled()


class HandlesSelf:
    """Mixin class that allows an object to act as an interpretation and its own handler.

    Note: this is unidiomatic for `effectful` code, but it simplifies our documentation and development process.

    In particular, it is not straightforward to define a decorator that automates interpretation handling whilst
    keeping IDE-friendly docstrings.
    """

    _cm = None

    def __enter__(self):
        self._cm = handler(self)
        self._cm.__enter__()
        return self._cm

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)


class Discretizer(ObjectInterpretation, HandlesSelf):
    """
    Discretize a continuous-time state evolution to a discrete-time state evolution.
    Args:
        discretize: Callable (CTSE) -> DTSE. Defaults to euler_maruyama.
    """

    def __init__(self, discretize=euler_maruyama):
        super().__init__()
        self.discretize = discretize

    @implements(sample)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context | None = None,
    ) -> FunctionOfTime:
        if isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            discrete_evolution = self.discretize(dynamics.state_evolution)
            dynamics = DynamicalModel(
                initial_condition=dynamics.initial_condition,
                state_evolution=discrete_evolution,
                observation_model=dynamics.observation_model,
                control_model=dynamics.control_model,
                state_dim=dynamics.state_dim,
                observation_dim=dynamics.observation_dim,
                control_dim=dynamics.control_dim,
            )
        return fwd(name, dynamics, context)


class Condition(ObjectInterpretation, HandlesSelf):
    def __init__(self, context: Context):
        super().__init__()
        self.context = context

    @implements(sample)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context | None = None,
    ) -> FunctionOfTime:
        # Ignore any context passed in the call and use the handler's context
        site_ctx = self.context
        return fwd(name, dynamics, site_ctx)


class BaseSimulator(ObjectInterpretation, HandlesSelf):
    """Base class for simulators/unrollers.

    Concrete simulators implement `simulate(context, dynamics)` and optionally
    override `add_solved_sites` if they need custom behavior.
    """

    @implements(sample)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ) -> FunctionOfTime:
        self.add_solved_sites(name, dynamics, context)
        return fwd(name, dynamics, context)

    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Only simulate if we have observation times
        if context is None or context.observations.times is None:
            return

        # Run the simulator
        simulated = self.simulate(context, dynamics)

        # Add the results from the simulator as deterministic sites
        for site_name, trajectory in simulated.items():
            numpyro.deterministic(site_name, trajectory)

    def simulate(self, context: Context, dynamics: DynamicalModel) -> dict[str, State]:
        """
        Args:
            context (Context): Context containing times and potentially controls.
            dynamics (DynamicalModel): The dynamical model to simulate.
        Returns:
            dict[str, Trajectory]: A dictionary mapping site names to simulated trajectories.
        """
        raise NotImplementedError()


class BaseCDDynamaxLogFactorAdder(ObjectInterpretation, HandlesSelf):
    @implements(sample)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ) -> FunctionOfTime:
        self.add_log_factors(name, dynamics, context)

        # Forward unchanged so downstream handlers (or default implementation)
        # can still see this op if needed.
        return fwd(name, dynamics, context)

    def add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Inheritors should implement this method.
        raise NotImplementedError()
