# handlers.py
"""Handlers for dsx operations using Interpretation-based style."""

from typing import TypeVar

from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from effectful.ops.types import NotHandled

from dynestyx.models import (
    DynamicalModel,
)
from dynestyx.types import FunctionOfTime

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
