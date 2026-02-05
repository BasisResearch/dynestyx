# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

import warnings

import numpyro
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements
from numpyro.primitives import Message

from dynestyx.discretizers import euler_maruyama
from dynestyx.dynamical_models import ContinuousTimeStateEvolution, DynamicalModel
from dynestyx.ops import Context, FunctionOfTime, States, sample_ds
from dynestyx.utils import HandlesSelf


class Discretizer(ObjectInterpretation, HandlesSelf):
    """
    Discretize a continuous-time state evolution to a discrete-time state evolution.
    Args:
        discretize: Callable (CTSE) -> DTSE. Defaults to euler_maruyama.
    """

    def __init__(self, discretize=euler_maruyama):
        super().__init__()
        self.discretize = discretize

    @implements(sample_ds)
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

    @implements(sample_ds)
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


class plate(numpyro.primitives.plate, ObjectInterpretation, HandlesSelf):
    """
    Wrapper around a `numpyro.primitives.plate` primitive.
    """

    @implements(sample_ds)
    def _sample_ds(
        self, name: str, dynamics: DynamicalModel, context: Context | None = None
    ) -> FunctionOfTime:
        fwd(name, dynamics, context)

    def process_message(self, msg: Message) -> None:
        if msg["type"] not in ("param", "sample", "plate", "deterministic"):
            if msg["type"] == "control_flow":
                warnings.warn(
                    "numpyro cannot use control flow primitives under a `plate` primitive. "
                    "There are internal reasons why this may occur in dsx, but you should not do this."
                )
            return
        try:
            return super().process_message(msg)
        except NotImplementedError as e:
            if "Cannot use control flow primitive under a `plate` primitive." in str(e):
                return
            raise e

    @property  # type: ignore[misc]
    def __class__(self):
        return numpyro.primitives.plate
