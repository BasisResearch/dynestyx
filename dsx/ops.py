from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled, Term
from dsx.dynamical_models import DynamicalModel
import jax
from typing import Callable, Optional

# Type alias for states: dict mapping state names to arrays
States = dict[str, jax.Array]
Times = jax.Array
FunctionOfTime = Callable[[Times], States]
Trajectory = tuple[Times, States]


@defop
def sample_ds(
        name: str,
        dynamics: DynamicalModel,
        obs: Optional[Trajectory]
    ) -> FunctionOfTime:

    raise NotHandled()
    