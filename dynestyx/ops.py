"""Shared effectful ops and handler mixins for dynestyx."""

from collections.abc import Callable

from effectful.ops.semantics import handler
from effectful.ops.syntax import defop
from jaxtyping import Array, Real

from dynestyx.models import DynamicalModel
from dynestyx.types import FunctionOfTime

_default_sample_interpretation: Callable[..., FunctionOfTime] | None = None


def set_default_sample_interpretation(
    fn: Callable[..., FunctionOfTime],
) -> None:
    """Register the fallback used when no handler consumes ``dsx.sample``.

    We keep the fallback outside the effectful operation object itself so handler
    registrations stay stable after class creation.
    """
    global _default_sample_interpretation
    _default_sample_interpretation = fn


@defop
def _sample_intp(
    name: str,
    dynamics: DynamicalModel,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None = None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
    **kwargs,
) -> FunctionOfTime:
    """Effectful sample op interpreted by active dynestyx handlers."""
    if _default_sample_interpretation is None:
        raise RuntimeError(
            "_sample_intp default rule is not configured. Import dynestyx.handlers "
            "before calling dynestyx ops directly."
        )
    return _default_sample_interpretation(
        name,
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
        **kwargs,
    )


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
        assert self._cm is not None
        return self._cm.__exit__(exc_type, exc, tb)
