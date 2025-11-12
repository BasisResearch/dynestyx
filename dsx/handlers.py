"""Handlers for dsx operations using Interpretation-based style."""

from dsx.ops import sample_ds, States, Times, FunctionOfTime, Trajectory
from dsx.dynamical_models import DynamicalModel
from typing import Callable, Optional
from effectful.ops.syntax import ObjectInterpretation, implements, Term
import jax
from effectful.ops.semantics import fwd


class Condition(ObjectInterpretation):

    def __init__(self, data: dict[str, Trajectory]) -> None:
        super().__init__()

        self.data = data

    @implements(sample_ds)
    def _sample_ds(
            self,
            name: str,
            dynamics: DynamicalModel,
            obs: Optional[Trajectory]
        ) -> FunctionOfTime:

        obs = self.data.get(name, None)
        if obs is None:
            return fwd()
        
        return fwd(name, dynamics, obs)


class BaseCDDynamaxLogFactorAdder(ObjectInterpretation):

    @implements(sample_ds)
    def _sample_ds(
            self,
            name: str,
            dynamics: DynamicalModel,
            obs: Optional[Trajectory]
        ) -> FunctionOfTime:

        if obs is not None:
            self.add_log_factors(name, dynamics, obs)
        
        return fwd()
        
    def add_log_factors(self, name: str, dynamics: DynamicalModel, obs: Optional[Trajectory]):

        # Inheritors should implement this method.
        raise NotImplementedError()


class BaseSolver(ObjectInterpretation):

    @implements(sample_ds)
    def _sample_ds(
            self,
            name: str,
            dynamics: DynamicalModel,
            obs: Optional[Trajectory]
        ) -> FunctionOfTime:

        # TODO consider instantiating a fresh defop with a free time variable?

        return lambda times: self.solve(times, dynamics)
    
    def solve(self, times: jax.Array, dynamics: DynamicalModel) -> States:
        
        # Inheritors should implement this method.
        raise NotImplementedError()
