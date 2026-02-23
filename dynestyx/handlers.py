# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

from typing import TypeVar

import numpyro
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import NotHandled

from dynestyx.discretizers import euler_maruyama
from dynestyx.models import (
    ContinuousTimeStateEvolution,
    DynamicalModel,
)
from dynestyx.types import FunctionOfTime, State

T = TypeVar("T")


@defop
def sample(
    name: str,
    dynamics: DynamicalModel,
    *,
    obs_times=None,
    obs_values=None,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
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
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
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
        return fwd(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )


class BaseSimulator(ObjectInterpretation, HandlesSelf):
    """Base class for simulators/unrollers.

    Concrete simulators implement `simulate(dynamics, obs_times, ...)` and optionally
    override `add_solved_sites` if they need custom behavior.
    """

    @implements(sample)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> FunctionOfTime:
        self.add_solved_sites(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
        return fwd(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

    def add_solved_sites(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ):
        # Only simulate if we have observation times
        if obs_times is None:
            return

        # Run the simulator
        simulated = self.simulate(
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

        # Add the results from the simulator as deterministic sites
        for site_name, trajectory in simulated.items():
            numpyro.deterministic(site_name, trajectory)

    def simulate(
        self,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> dict[str, State]:
        """
        Args:
            dynamics: The dynamical model to simulate.
            obs_times: Observation times.
            obs_values: Observed values (optional).
            ctrl_times: Control times (optional).
            ctrl_values: Control values (optional).
        Returns:
            dict[str, State]: A dictionary mapping site names to simulated trajectories.
        """
        raise NotImplementedError()


class BaseCDDynamaxLogFactorAdder(ObjectInterpretation, HandlesSelf):
    """Base for filter handlers."""

    @implements(sample)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> FunctionOfTime:
        self.add_log_factors(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
        # Forward unchanged so downstream handlers (or default implementation)
        # can still see this op if needed.
        return fwd(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

    def add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ):
        # Inheritors should implement this method.
        raise NotImplementedError()
