"""Contains the `sample` and `infer` primitives and `effectful` utilities for `dynestyx`."""

from typing import TypeVar

import numpyro
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import NotHandled
from jaxtyping import Array, Real

from dynestyx.models import (
    DynamicalModel,
)
from dynestyx.types import ConditionedResult, FunctionOfTime
from dynestyx.utils import (
    _get_dynamics_with_t0,
    _validate_control_dim,
    _validate_controls,
    _validate_site_sorting,
)

T = TypeVar("T")


def _validate_and_prepare(
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
) -> DynamicalModel:
    """Validate inputs and return dynamics with t0 resolved."""
    if obs_times is None and predict_times is None:
        raise ValueError("At least one of obs_times or predict_times must be provided")

    if (obs_times is None and obs_values is not None) or (
        obs_times is not None and obs_values is None
    ):
        raise ValueError(
            "obs_times and obs_values must be provided together, or both None"
        )

    if obs_times is not None and obs_values is not None:
        obs_T = obs_times.shape[-1]
        if obs_values.ndim == 1:
            val_T = len(obs_values)
        else:
            if obs_values.shape[-1] == obs_T:
                val_T = obs_values.shape[-1]
            elif obs_values.ndim >= 2 and obs_values.shape[-2] == obs_T:
                val_T = obs_values.shape[-2]
            else:
                val_T = obs_values.shape[-1]
        if obs_T != val_T:
            raise ValueError(
                f"obs_times and obs_values must have the same number of time steps. "
                f"Got obs_times time dim={obs_T}, obs_values time dim={val_T}."
            )

    _validate_site_sorting(obs_times, name="obs_times")
    _validate_site_sorting(ctrl_times, name="ctrl_times")
    _validate_site_sorting(predict_times, name="predict_times")

    _validate_controls(obs_times, predict_times, ctrl_times, ctrl_values)
    _validate_control_dim(dynamics, ctrl_values)

    return _get_dynamics_with_t0(dynamics, obs_times, predict_times)


def condition(
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
):
    """Run inference on a dynamical model without registering numpyro sites.

    This is the numpyro-free entry point. When a Filter or Smoother handler
    is active, returns a ConditionedResult dataclass with marginal_loglik, states, etc.
    """
    dynamics_with_t0 = _validate_and_prepare(
        name,
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
    )

    return _condition_intp(
        name,
        dynamics_with_t0,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
        **kwargs,
    )


def sample(
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
):
    """
    Samples from a dynamical model. This is the main primitive of dynestyx.

    The ``sample`` primitive is meant to mimic the ``numpyro.sample`` primitive
    in usage, but using a ``DynamicalModel`` instead of a ``Distribution``.

    Internally, ``sample`` calls ``dsx.condition(...)`` and then registers the
    results as numpyro sites (``numpyro.factor``, ``numpyro.deterministic``).
    """
    result = condition(
        name,
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        predict_times=predict_times,
        **kwargs,
    )

    if (
        isinstance(result, ConditionedResult)
        and result._register_numpyro_sites is not None
    ):
        result._register_numpyro_sites(name)

    return result


@defop
def _condition_intp(
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
    """The functional version of `sample` to be interpreted at runtime."""
    raise NotHandled()


class HandlesSelf:
    """Mixin class that allows an object to act as an interpretation and its own handler."""

    _cm = None

    def __enter__(self):
        self._cm = handler(self)
        self._cm.__enter__()
        return self._cm

    def __exit__(self, exc_type, exc, tb):
        assert self._cm is not None
        return self._cm.__exit__(exc_type, exc, tb)


class plate(ObjectInterpretation):
    """Hierarchical plate for batched trajectories."""

    def __init__(self, name: str, size: int, dim=None):
        if dim is not None:
            raise NotImplementedError(
                "The `dim` argument is not currently supported for dynestyx plates"
            )

        self.name = name
        self.size = size
        self._numpyro_plate = numpyro.plate(name, size, dim=dim)
        self._cm = None

    def __enter__(self):
        """Enter both numpyro.plate context and dynestyx plate interpretation."""
        self._numpyro_plate.__enter__()
        self._cm = handler(self)
        self._cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit both numpyro.plate context and dynestyx plate interpretation."""
        self._cm.__exit__(exc_type, exc, tb)
        return self._numpyro_plate.__exit__(exc_type, exc, tb)

    @implements(_condition_intp)
    def _sample_ds(
        self, name, dynamics, *, plate_shapes=(), **kwargs
    ) -> FunctionOfTime:
        """Effectful interpretation for the `sample` primitive in a plate."""
        return fwd(name, dynamics, plate_shapes=plate_shapes + (self.size,), **kwargs)
