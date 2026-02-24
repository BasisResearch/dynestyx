"""Contains the `sample` primitive and `effectful` utilities for `dynestyx`."""

from typing import TypeVar

import jax
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
    obs_times: jax.Array,
    obs_values: jax.Array | None = None,
    ctrl_times: jax.Array | None = None,
    ctrl_values: jax.Array | None = None,
    **kwargs,
) -> FunctionOfTime:
    """
    Samples from a dynamical model. This is the main primitive of dynestyx.

    The `sample` primitive is meant to mimic the `numpyro.sample` primitive in usage,
    but using a `DynamicalModel` instead of a `Distribution`.

    The `sample` method is defined as a `defop` in `effectful`, meaning that
    it is an undefined function here, but "interpreted" at runtime. In other words,
    the actual implementation of a `sample` operation is determined by the context
    in which it is used, e.g., within a `Filter` or `Simulator` object.

    Parameters:
        name: Name of the sample site.
        dynamics: Dynamical model to sample from.
        obs_times: Times at which to sample the observations.
        obs_values: Values of the observations at the given times.
        ctrl_times: Times at which to sample the controls.
        ctrl_values: Values of the controls at the given times.
        **kwargs: Additional keyword arguments.

    Returns:
        FunctionOfTime: A function of time that samples from the dynamical model.
    """
    raise NotHandled()


class HandlesSelf:
    """Mixin class that allows an object to act as an interpretation and its own handler.

    This is used by most inference and simulation objects in `dynestyx`, allowing them to provide
    an object interpretation of the `sample` operation whilst still being used directly as a handler.

    ??? note
        This is unidiomatic for `effectful` code, but it simplifies our documentation and development process.

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
