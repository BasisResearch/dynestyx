import dataclasses
from collections.abc import Callable

from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled
from jax import Array

from dynestyx.dynamical_models import DynamicalModel

# Type alias for states: dict mapping state names to arrays, or just an array
Times = Array
States = dict[str, Array] | Array
FunctionOfTime = Callable[[Times], States]


@dataclasses.dataclass
class Trajectory:
    """
    A 1D time axis and values living on that axis.

    Semantics:
      - times is None  -> times are implicit / inferred / shared with some other grid
      - values is None -> "no values here" (e.g. just a solve grid)
    """

    times: Times | None = None
    values: States | None = None


@dataclasses.dataclass
class Context:
    """
    All time-indexed info for a single sample site.
    """

    # Where to solve the dynamics
    solve: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Observations
    observations: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Controls u(t), if any
    controls: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Extensible: extra time-indexed series or metadata
    extras: dict[str, Trajectory] = dataclasses.field(default_factory=dict)


@defop
def sample(
    name: str, dynamics: DynamicalModel, context: Context | None = None
) -> FunctionOfTime:
    raise NotHandled()
