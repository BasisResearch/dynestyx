"""Handlers for dsx operations using Interpretation-based style."""

from dsx.ops import sample_ds, States, Times, FunctionOfTime, Trajectory
from dsx.dynamical_models import DynamicalModel
from typing import Callable, Optional
from effectful.ops.syntax import ObjectInterpretation, implements, Term
from effectful.ops.semantics import fwd
import numpyro

class Condition(ObjectInterpretation):

    def __init__(self, data: dict[str, Trajectory]) -> None:
        super().__init__()

        self.data = data

    @implements(sample_ds)
    def _sample_ds(
            self,
            dynamics: DynamicalModel,
            obs: Optional[Trajectory],
            name: Optional[str] = None
        ) -> FunctionOfTime:

        obs = self.data.get(name, None)
        if obs is None:
            return fwd()
        
        return fwd(dynamics, obs, name)


class BaseCDDynamaxLogFactorAdder(ObjectInterpretation):

    @implements(sample_ds)
    def _sample_ds(
            self,
            dynamics: DynamicalModel,
            obs: Optional[Trajectory],
            name: Optional[str] = None
        ) -> FunctionOfTime:

        if obs is not None:
            self.add_log_factors(dynamics, obs, name)
        
        return fwd()
        
    def add_log_factors(self, dynamics: DynamicalModel, obs: Optional[Trajectory], name: Optional[str] = None):

        # Inheritors should implement this method.
        raise NotImplementedError()


class BaseSolver(ObjectInterpretation):

    @implements(sample_ds)
    def _sample_ds(
            self,
            dynamics: DynamicalModel,
            times: Optional[Trajectory],
            name: Optional[str] = None
        ) -> FunctionOfTime:

        self.add_solved_sites(dynamics, times, name)
        
        return fwd()
    
    def solve(self, times: Times, dynamics: DynamicalModel) -> dict[str, Trajectory]:
        """
        Args:
            times (Times): Array of times at which to solve the dynamics.
            dynamics (DynamicalModel): The dynamical model to solve.
        Returns:
            dict[str, Trajectory]: A dictionary mapping site names to solved trajectories.
        """
        raise NotImplementedError()
    
    def add_solved_sites(self, dynamics: DynamicalModel, times: Times, name: Optional[str] = None):

        # Run the solver
        new_sites = self.solve(times, dynamics)
        
        # Add the results from the solver as deterministic sites
        for site_name, trajectory in new_sites.items():
            numpyro.deterministic(site_name, trajectory)
    