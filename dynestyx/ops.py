import dataclasses
import warnings
from collections.abc import Callable

import jax
import numpyro
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled
from jax import Array
from numpyro.primitives import (
    Message,
)

from dynestyx.dynamical_models import DynamicalModel

# Type alias for states: dict mapping state names to arrays, or just an array
Times = Array
States = dict[str, Array] | Array
FunctionOfTime = Callable[[Times], States]


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Trajectory:
    times: Times | None = None
    values: States | None = None

    def tree_flatten(self):
        # None is allowed as a leaf; JAX treats it as static-ish leaf
        return (self.times, self.values), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        times, values = children
        return cls(times=times, values=values)


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class Context:
    solve: Trajectory = dataclasses.field(default_factory=Trajectory)
    observations: Trajectory = dataclasses.field(default_factory=Trajectory)
    controls: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Extensible: extra time-indexed series or metadata
    extras: dict[str, Trajectory] = dataclasses.field(default_factory=dict)

    def tree_flatten(self):
        return (self.solve, self.observations, self.controls, self.extras), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        solve, observations, controls, extras = children
        return cls(
            solve=solve, observations=observations, controls=controls, extras=extras
        )


@defop
def sample_ds(
    name: str, dynamics: DynamicalModel, context: Context | None = None
) -> FunctionOfTime:
    raise NotHandled()


class plate(numpyro.primitives.plate):
    """
    Wrapper around a `numpyro.primitives.plate` primitive.
    """

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
