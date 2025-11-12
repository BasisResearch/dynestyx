import pytest
import jax.numpy as jnp
from effectful.ops.semantics import handler
from dsx.handlers import BaseSolver, States
from dsx.dynamical_models import ContinuousTimeDynamicalModel
from dsx.ops import sample_ds
from typing import Callable


class DumbDynamics(ContinuousTimeDynamicalModel):
    """Dynamical model that takes a function of time directly."""
    
    def __init__(self, func: Callable, state_name: str = "x"):
        self.func = func
        self.state_name = state_name


class DumbSolver(BaseSolver):
    """Solver that only works with DumbDynamics."""
    
    def solve(self, times, dynamics: DumbDynamics) -> States:
        if not isinstance(dynamics, DumbDynamics):
            raise NotImplementedError(f"DumbSolver only works with DumbDynamics, got {type(dynamics)}")
        return {dynamics.state_name: dynamics.func(times)}


def test_sin_solver():
    """Test that solver returns sin(t)."""
    solver = handler(DumbSolver())
    dynamics = DumbDynamics(jnp.sin, state_name="x")
    
    # Sample the trajectory within handler context
    with solver:
        trajectory_fn = sample_ds("f", dynamics, None)
    
    # Evaluate at some times
    times = jnp.array([0.0, jnp.pi/2, jnp.pi])
    result = trajectory_fn(times)
    
    # Check we get sin(t) back
    expected = jnp.sin(times)
    assert jnp.allclose(result["x"], expected), \
        f"Expected sin(t), got {result['x']}"

