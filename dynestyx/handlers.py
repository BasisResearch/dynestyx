"""Contains the `sample` primitive and `effectful` utilities for `dynestyx`."""

from typing import TypeVar

import numpyro
from effectful.ops.semantics import fwd, handler
from effectful.ops.syntax import ObjectInterpretation, defop, implements
from effectful.ops.types import NotHandled
from jaxtyping import Array, Bool, Real

from dynestyx.models import (
    DynamicalModel,
)
from dynestyx.observation_missingness import prepare_observation_views
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
    """
    Samples from a dynamical model. This is the main primitive of dynestyx.

    The `sample` primitive is meant to mimic the `numpyro.sample` primitive in usage,
    but using a `DynamicalModel` instead of a `Distribution`.

    The `sample` method calls `_sample_intp`, which is defined as a `defop` in `effectful`.
    This is where any real "work" is done, after input validation.

    Shape note:
        Inside ``dsx.plate``, observation arrays use leading plate axes followed
        by time and event axes, e.g. ``(N, T, obs_dim)``. Model parameters follow
        the same leading-plate, trailing-event convention. See :class:`plate`
        for the full plated-shape contract.

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
    obs_values_filled, obs_mask, obs_has_missing = prepare_observation_views(
        dynamics_with_t0, obs_values
    )

    # Pass to interpreted version of `sample` for inference.
    return _sample_intp(
        name,
        dynamics_with_t0,
        obs_times=obs_times,
        obs_values=obs_values,
        _obs_values_filled=obs_values_filled,
        _obs_mask=obs_mask,
        _obs_has_missing=obs_has_missing,
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
    obs_times: Real[Array, "*obs_time_plate obs_time"] | None = None,
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"]
    | None = None,
    _obs_values_filled: Array | None = None,
    _obs_mask: Bool[Array, "*obs_value_plate obs_time observation_dim"]
    | Bool[Array, "*obs_value_plate obs_time"]
    | None = None,
    _obs_has_missing: bool | None = None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate ctrl_time control_dim"]
    | Real[Array, "*ctrl_value_plate ctrl_time"]
    | None = None,
    predict_times: Real[Array, "*predict_time_plate predict_time"] | None = None,
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
        _obs_values_filled: Internal mask-aware version of ``obs_values`` with
            missing entries replaced by neutral fillers while preserving shape.
        _obs_mask: Internal boolean mask marking which observation entries are
            truly observed.
        _obs_has_missing: Internal precomputed flag indicating whether any
            observation entries are missing.
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
        assert self._cm is not None
        return self._cm.__exit__(exc_type, exc, tb)


class plate(ObjectInterpretation):
    """Hierarchical plate for batched trajectories.

    ``dsx.plate`` wraps ``numpyro.plate`` for parameter sampling semantics and
    intercepts ``dsx.sample`` to pass plate sizes to simulator and filter
    handlers. Use it when a dynamical system has conditionally independent
    members, such as multiple trajectories, patients, groups, or treatment
    arms.

    Shape semantics:
        Dynestyx treats plate axes as leading data-batch axes. Time axes come
        after plate axes in observation arrays, and state/observation event axes
        remain trailing axes.

        For one plate of size ``N``:

        ```python
        obs_values          # (N, T, obs_dim), or (N, T) for scalar observations
        mu_i                # (N, state_dim), a vector parameter per trajectory
        initial_mean        # (N, state_dim), or shared as (state_dim,)
        initial_cov         # (N, state_dim, state_dim), or shared as (state_dim, state_dim)
        prior["f_states"]   # (num_samples, N, n_sim, T, state_dim)
        ```

        Distribution-valued model components use their NumPyro
        ``batch_shape``/``event_shape`` split: leading plate dimensions are
        batch dimensions, while state and observation sizes are inferred from
        ``event_shape``. Thus a batched initial condition may have
        ``loc.shape == (N, state_dim)`` with either shared or batched covariance.

        Built-in LTI vector fields, including transition/drift and observation
        biases, may be shared or plate-batched:

        ```python
        with dsx.plate("trajectories", N):
            mu_i = numpyro.sample(
                "mu_i",
                dist.Normal(mu_global, sigma).to_event(1),
            )  # (N, state_dim)

            dynamics = LTI_discrete(
                A=A,
                Q=Q,
                H=H,
                R=R,
                b=mu_i,              # plate-batched vector bias
                initial_mean=mu_i,   # plate-batched initial mean
            )
        ```

        Ambiguous arrays are kept shared rather than sliced. In particular, a
        shared vector whose length happens to equal ``N`` is not treated as
        plate-batched unless it is a known vector-valued model field with an
        explicit event axis, such as ``(N, state_dim)``. For one-dimensional
        vector fields, prefer explicit singleton event axes like ``(N, 1)``.
        Nested plates follow the same rule with multiple leading plate axes.

    Why event shapes drive sizing:
        Inside a plate, ``state_dim`` and ``observation_dim`` are inferred from
        a distribution's NumPyro ``event_shape``, not from the full sample
        shape. The full sample shape includes leading plate batch axes, which
        are independent-member dimensions, not event dimensions; using it
        would misread ``(N, d)`` as ``state_dim == N``. Sticking to
        ``event_shape`` keeps the per-member event size unambiguous.

        The contract for a single plate of size ``N``:

        | Sampled shape   | event_shape   | Interpretation                   |
        | --------------- | ------------- | -------------------------------- |
        | ``(d,)``        | ``(d,)``      | Shared vector event, broadcast.  |
        | ``(N, d)``      | ``(d,)``      | Per-member vector event of dim d.|
        | ``(N,)``        | ``()``        | Per-member **scalar** event.     |

        The third row is the subtle one: ``dist.Normal(mu, sigma)`` with
        ``mu.shape == (N,)`` produces ``event_shape == ()``, which we treat as
        ``state_dim == 1``. If the intent is a 1-D *vector* state with one
        entry per member, wrap with ``.to_event(1)`` or use a vector-valued
        distribution (``dist.MultivariateNormal``) so the rank-1 axis is an
        event axis. This is the same ambiguity rule as the shared-vector case
        above, applied at the distribution level.

        Nested plates extend this with multiple leading batch axes; the inner
        plate is the leftmost data batch axis, matching NumPyro's convention.

    Output axis ordering:
        Predictive draws and filter outputs preserve a consistent axis order:

        ``(num_samples, *plate_axes_inner_to_outer, n_sim, T, *event_shape)``

        For example, with one plate of size ``N`` and a vector state:

        ```python
        prior["f_states"]        # (num_samples, N, n_sim, T, state_dim)
        prior["f_observations"]  # (num_samples, N, n_sim, T, obs_dim)
        ```

        ``num_samples`` comes first (NumPyro ``Predictive``), then plate axes
        from inner to outer (the inner plate is the leftmost data batch axis,
        so it appears immediately after ``num_samples``), then ``n_sim`` from
        the simulator, then time, then the event axes. ``flatten_draws`` is
        the standard helper for collapsing ``(num_samples, n_sim)`` for
        plotting/credible intervals.

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

    Sharp edges:
        - **Drift/diffusion must be sliceable pytrees.** Plated parameters must be
          stored as array fields of an ``eqx.Module``, not captured in a Python
          closure. A closure-captured variable is invisible to pytree munging,
          and can introduce shape errors. The built-in components (``AffineDrift``,
          ``LTI_continuous``, ``FullDiffusion``, etc.) follow this rule. See the
          hierarchical inference tutorial (``08_hierarchical_inference.ipynb``) for
          the full sharp-edges list including event-shape vs. sample-shape rules and
          the rank-1 shared/batched ambiguity.

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
