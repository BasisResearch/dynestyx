from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled
from dsx.dynamical_models import DynamicalModel
from typing import Dict, Callable, Optional
from jax import Array
import dataclasses

# Type alias for states: dict mapping state names to arrays
Times = Array
States = Array #[str, Array]
FunctionOfTime = Callable[[Times], States]

@dataclasses.dataclass
class Trajectory:
    """
    A 1D time axis and values living on that axis.

    Semantics:
      - times is None  -> times are implicit / inferred / shared with some other grid
      - values is None -> "no values here" (e.g. just a solve grid)
    """

    times: Optional[Times] = None
    values: Optional[States] = None

@dataclasses.dataclass
class Context:
    """
    All time-indexed info for a single sample_ds site.
    """

    # Where to solve the dynamics
    solve: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Observations
    observations: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Controls u(t), if any
    controls: Trajectory = dataclasses.field(default_factory=Trajectory)

    # Extensible: extra time-indexed series or metadata
    extras: Dict[str, Trajectory] = dataclasses.field(default_factory=dict)


@defop
def sample_ds(name: str,
              dynamics: DynamicalModel,
              context: Optional[Context] = None) -> FunctionOfTime:
    raise NotHandled()
    