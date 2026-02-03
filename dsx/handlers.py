# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

from typing import Optional

from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.semantics import fwd
import numpyro

from dsx.ops import sample_ds, FunctionOfTime, Context, States
from dsx.dynamical_models import DynamicalModel, ContinuousTimeStateEvolution
from dsx.discretizers import EulerMaruyamaDiscretization

from effectful.ops.semantics import handler


class HandlesSelf:
    _cm = None

    def __enter__(self):
        self._cm = handler(self)
        self._cm.__enter__()
        return self._cm

    def __exit__(self, exc_type, exc, tb):
        return self._cm.__exit__(exc_type, exc, tb)


class Discretizer(ObjectInterpretation, HandlesSelf):
    """
    Base class for discretizing continuous-time state evolution.
    Subclasses implement discretize_state_evolution().
    """

    def __init__(self, scheme: type = EulerMaruyamaDiscretization):
        super().__init__()
        self.scheme = scheme

    @implements(sample_ds)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Optional[Context] = None,
    ) -> FunctionOfTime:
        if isinstance(dynamics.state_evolution, ContinuousTimeStateEvolution):
            discrete_evolution = self.scheme(
                continuous_time_evolution=dynamics.state_evolution
            )
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

    @implements(sample_ds)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Optional[Context] = None,
    ) -> FunctionOfTime:
        # Ignore any context passed in the call and use the handler's context
        site_ctx = self.context
        return fwd(name, dynamics, site_ctx)


class BaseSimulator(ObjectInterpretation, HandlesSelf):
    """Base class for simulators/unrollers.

    Concrete simulators implement `simulate(context, dynamics)` and optionally
    override `add_solved_sites` if they need custom behavior.
    """

    @implements(sample_ds)
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
        if isinstance(simulated, dict):
            for site_name, trajectory in simulated.items():
                # TODO: dw: check this type ignore. I think it has a point...
                numpyro.deterministic(site_name, trajectory)  # type: ignore
        else:
            # If it's just an array (shouldn't happen for simulate() but handle it)
            numpyro.deterministic("value", simulated)

    def simulate(self, context: Context, dynamics: DynamicalModel) -> States:
        """
        Args:
            context (Context): Context containing times and potentially controls.
            dynamics (DynamicalModel): The dynamical model to simulate.
        Returns:
            dict[str, Trajectory]: A dictionary mapping site names to simulated trajectories.
        """
        raise NotImplementedError()


class BaseCDDynamaxLogFactorAdder(ObjectInterpretation, HandlesSelf):
    @implements(sample_ds)
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
