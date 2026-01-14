# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

from typing import Optional

from effectful.ops.syntax import ObjectInterpretation, implements
from effectful.ops.semantics import fwd
import numpyro

from dsx.ops import sample_ds, Times, FunctionOfTime, Context, States
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
        context: Optional[Context] = None,
    ) -> FunctionOfTime:
        # Ignore any context passed in the call and use the handler's context
        site_ctx = self.context
        return fwd(name, dynamics, site_ctx)


class BaseUnroller(ObjectInterpretation):
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
        raise NotImplementedError()


class BaseSolver(BaseUnroller):
    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        context: Context,
    ):
        # Only solve if we have solve-times
        if context is None or context.observations.times is None:
            return

        # Run the solver
        # Make sure this can throw an error if needed? I think it is not.
        new_sites = self.solve(context.observations.times, dynamics)

        # Add the results from the solver as deterministic sites
        # solve() always returns Dict[str, Array], but States is Union for Trajectory.values
        if isinstance(new_sites, dict):
            for site_name, trajectory in new_sites.items():
                # TODO: dw: check this type ignore. I think it has a point...
                numpyro.deterministic(site_name, trajectory)  # type: ignore
        else:
            # If it's just an array (shouldn't happen for solve() but handle it)
            numpyro.deterministic("value", new_sites)

    def solve(self, times: Times, dynamics: DynamicalModel) -> States:
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
