# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from functools import wraps
from typing import Any, TypeVar

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


def handles[T](
    cls: type[T],
) -> Callable[[Callable[..., Any]], Callable[..., AbstractContextManager[Any]]]:
    """
    @handles(SomeClass)
    def f(...): ...

    Then: with f(*args, **kwargs): ...
    will do: handle(SomeClass(*args, **kwargs))
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., AbstractContextManager[Any]]:
        @wraps(fn)
        def wrapped(*args: Any, **kwargs: Any) -> AbstractContextManager[Any]:
            @contextmanager
            def cm():
                obj = cls(*args, **kwargs)
                with handler(obj):
                    yield

            return cm()

        return wrapped

    return decorator


class DiscretizerObjIntp(ObjectInterpretation):
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


@handles(DiscretizerObjIntp)
def Discretizer(  # type: ignore[empty-body]
    name: str, dynamics: DynamicalModel, context: Context | None = None
) -> FunctionOfTime:
    pass


class ConditionObjIntp(ObjectInterpretation):
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


@handles(ConditionObjIntp)
def Condition(  # type: ignore[empty-body]
    name: str, dynamics: DynamicalModel, context: Context | None = None
) -> FunctionOfTime:
    pass


class BaseSimulatorObjIntp(ObjectInterpretation):
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


class BaseCDDynamaxLogFactorAdder(ObjectInterpretation):
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
