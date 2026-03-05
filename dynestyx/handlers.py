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
from dynestyx.utils import (
    _get_dynamics_with_t0,
    _validate_control_dim,
    _validate_controls,
    _validate_site_sorting,
)

T = TypeVar("T")


def sample(
    name: str,
    dynamics: DynamicalModel,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array | None = None,
    ctrl_times: jax.Array | None = None,
    ctrl_values: jax.Array | None = None,
    predict_times: jax.Array | None = None,
    **kwargs,
) -> FunctionOfTime:
    """
    Samples from a dynamical model. This is the main primitive of dynestyx.

    The `sample` primitive is meant to mimic the `numpyro.sample` primitive in usage,
    but using a `DynamicalModel` instead of a `Distribution`.

    The `sample` method calls `_sample_intp`, which is defined as a `defop` in `effectful`.
    This is where any real "work" is done, after input validation.

    Parameters:
        name: Name of the sample site.
        dynamics: Dynamical model to sample from.
        obs_times: Times at which to sample the observations.
        obs_values: Values of the observations at the given times.
        ctrl_times: Times at which to sample the controls.
        ctrl_values: Values of the controls at the given times.
        predict_times: Times at which to predict the observations.
        **kwargs: Additional keyword arguments.

    Returns:
        FunctionOfTime: A function of time that samples from the dynamical model.
    """
    # Rule: obs_times must be accompanied with obs_values, which should be the same length.
    if obs_times is None and predict_times is None:
        raise ValueError("At least one of obs_times or predict_times must be provided")

    if (obs_times is None and obs_values is not None) or (
        obs_times is not None and obs_values is None
    ):
        raise ValueError(
            "obs_times and obs_values must be provided together, or both None"
        )

    if (
        obs_times is not None
        and obs_values is not None
        and len(obs_times) != len(obs_values)
    ):
        raise ValueError("obs_times and obs_values must be the same length")

    _validate_site_sorting(obs_times, ctrl_times)
    _validate_controls(obs_times, predict_times, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)

    # Initial dynamics may not have t0, which is then inferred from obs_times
    dynamics_with_t0 = _get_dynamics_with_t0(dynamics, obs_times, predict_times)

    # Pass to interpreted version of `sample` for inference.
    return _sample_intp(
        name,
        dynamics_with_t0,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
        **kwargs,
    )


@defop
def _sample_intp(
    name: str,
    dynamics: DynamicalModel,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array | None = None,
    ctrl_times: jax.Array | None = None,
    ctrl_values: jax.Array | None = None,
    predict_times: jax.Array | None = None,
    **kwargs,
) -> FunctionOfTime:
    """
    The functional version of `sample` to be interpreted at runtime.

    This is implemented as a `defop` in `effectful`, meaning it is
    an undefined function here, but "interpreted" at runtime. In other words,
    the actual implementation of a `sample` operation is determined by the context
    in which it is used, e.g., within a `Filter` or `Simulator` object.

    Parameters:
        name: Name of the sample site.
        dynamics: Dynamical model to sample from.
        obs_times: Times at which to sample the observations.
        obs_values: Values of the observations at the given times.
        ctrl_times: Times at which to sample the controls.
        ctrl_values: Values of the controls at the given times.
        predict_times: Times at which to predict the observations.
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
