# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

from typing import Optional, Dict

from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.semantics import fwd
import numpyro

from dsx.ops import sample_ds, Times, FunctionOfTime, Trajectory, Context
from dsx.dynamical_models import DynamicalModel

class Condition(ObjectInterpretation):
    def __init__(self, context: Context):
        super().__init__()
        self.context = context

    @implements(sample_ds)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Optional[Context],
    ) -> FunctionOfTime:
        # Ignore any context passed in the call and use the handler's context
        site_ctx = self.context
        return fwd(name, dynamics, site_ctx)


class BaseSolver(ObjectInterpretation):

    @implements(sample_ds)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Optional[Context],
    ) -> FunctionOfTime:

        # Only solve if we have solve-times
        if context is not None and context.solve.times is not None:
            self.add_solved_sites(dynamics, context.solve.times, name)

        return fwd(name, dynamics, context)

    def add_solved_sites(
        self,
        dynamics: DynamicalModel,
        times: Times,
        name: Optional[str] = None,
    ):

        # Run the solver
        new_sites = self.solve(times, dynamics)

        # Add the results from the solver as deterministic sites
        for site_name, trajectory in new_sites.items():
            numpyro.deterministic(site_name, trajectory)

    def solve(self, times: Times, dynamics: DynamicalModel) -> Dict[str, Trajectory]:
        """
        Args:
            times (Times): Array of times at which to solve the dynamics.
            dynamics (DynamicalModel): The dynamical model to solve.
        Returns:
            dict[str, Trajectory]: A dictionary mapping site names to solved trajectories.
        """
        raise NotImplementedError()

class BaseCDDynamaxLogFactorAdder(ObjectInterpretation):

    @implements(sample_ds)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Optional[Context],
    ) -> FunctionOfTime:

        if context is not None:
            self.add_log_factors(dynamics, context, name)

        # Forward unchanged so downstream handlers (or default implementation)
        # can still see this op if needed.
        return fwd(name, dynamics, context)

    def add_log_factors(
        self,
        dynamics: DynamicalModel,
        context: Context,
        name: Optional[str] = None,
    ):
        # Inheritors should implement this method.
        raise NotImplementedError()
