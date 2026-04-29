"""Contains the `sample` primitive and `effectful` utilities for `dynestyx`."""

from typing import TypeVar

import jax
import numpyro
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
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
    obs_times: jax.Array | None = None,
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

    if obs_times is not None and obs_values is not None:
        # Compare time dimensions: obs_times is (..., T). obs_values can be
        # (..., T) for scalar observations or (..., T, D) for vector observations,
        # with optional leading plate dims.
        obs_T = obs_times.shape[-1]
        if obs_values.ndim == 1:
            val_T = len(obs_values)
        else:
            # Prefer a trailing time axis for scalar-observation tensors (..., T),
            # else fall back to the penultimate axis for (..., T, D).
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
    obs_times: jax.Array | None = None,
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


class plate(ObjectInterpretation):
    """Hierarchical plate for batched trajectories. Allows for multiple levels of hierarchy.

    Wraps ``numpyro.plate`` for parameter sampling semantics and intercepts
    ``dsx.sample`` to inject ``plate_shapes`` into the handler chain via ``fwd()``.
    Nested plate handlers run from inner to outer under ``effectful`` dispatch,
    so each handler appends its size to preserve NumPyro's effective batch order
    (inner plate is the leftmost data batch dim).

    Examples:
        >>> with dsx.plate("trajectories", M):
        ...     theta = numpyro.sample("theta", dist.Normal(0, 1))  # shape (M,)
        ...     dynamics = DynamicalModel(...)  # built from theta
        ...     dsx.sample("f", dynamics, obs_times=t, obs_values=y)

        >>> with dsx.plate("groups", G):
        ...     beta = numpyro.sample("beta", dist.Normal(0, 1))  # shape (G,)
        ...     with dsx.plate("trajectories", M):
        ...         alpha = numpyro.sample("alpha", dist.Normal(beta, 1))  # shape (M, G)
        ...         dynamics = DynamicalModel(...)  # built from alpha
        ...         dsx.sample("f", dynamics, obs_times=t, obs_values=y)

    Note:
        The `dim` argument is not currently supported for dynestyx plates.
    """

    def __init__(self, name: str, size: int, dim: int | None = None):
        """Initialize the plate handler.

        Parameters:
            name: Name of the plate.
            size: Size of the plate.
            dim: Dimension of the plate.
        """

        if dim is not None:
            raise ValueError(
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

    @implements(_sample_intp)
    def _sample_ds(
        self, name, dynamics, *, plate_shapes=(), **kwargs
    ) -> FunctionOfTime:
        """Effectful interpretation for the `sample` primitive in a plate.

        Appends metadata to the argument stack and passes forward.

        Parameters:
            name: Name of the sample site.
            dynamics: Dynamical model to sample from.
            plate_shapes: Shapes of plates (from plates that are more nested than this one).
            **kwargs: Additional keyword arguments.

        Returns:
            FunctionOfTime: A function of time that samples from the dynamical model.
        """
        # Append plate_shapes metadata to the argument stack and pass forward.
        return fwd(name, dynamics, plate_shapes=plate_shapes + (self.size,), **kwargs)
