"""Handlers for dsx operations using Interpretation-based style."""

from dsx.ops import sample_ds, States, Times, FunctionOfTime, Trajectory
from dsx.dynamical_models import DynamicalModel
from typing import Callable, Optional
from effectful.ops.syntax import ObjectInterpretation, implements, Term
import jax
from effectful.ops.semantics import fwd
from dsx import TODO


class Condition(ObjectInterpretation):

    def __init__(self, obs: tuple[Times, States]) -> None:
        super().__init__()

        self.obs = obs

    @implements(sample_ds)
    def _sample_ds(
            self,
            name: str,
            dynamics: DynamicalModel,
            obs: Optional[Trajectory]
        ) -> FunctionOfTime:
        
        fwd(name, dynamics, self.obs)


class BaseCDDynamaxLogFactorAdder(ObjectInterpretation):

    @implements(sample_ds)
    def _sample_ds(
            self,
            name: str,
            dynamics: DynamicalModel,
            obs: Optional[Trajectory]
        ) -> FunctionOfTime:

        if obs is None:
            fwd()
        
    def add_log_factors(self, dynamics: DynamicalModel, obs: Optional[Trajectory]):

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

        return lambda times: self.solve(times, dynamics)
    
    def solve(self, times: jax.Array, dynamics: DynamicalModel) -> States:
        
        # Inheritors should implement this method.
        raise NotImplementedError()
