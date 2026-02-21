"""Type aliases for dynamical systems."""

from collections.abc import Callable

import jax

State = jax.Array
dState = State
Observation = jax.Array
Control = State | None
Time = jax.Array
Params = dict[str, float | jax.Array]
Key = jax.Array
FunctionOfTime = Callable[[Time], State]
