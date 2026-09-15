"""State evolution for states that live on one or more spatial fields.

`dynestyx` states are always a flat, rank-1 vector. A PDE solver, by contrast, works on
fields with shape `(n_components, *spatial_shape)` -- a scalar vorticity field
`(1, 128, 128)`, a 3D velocity field `(3, 64, 64, 64)`, or several coupled fields of
different sizes. `SpatialStateEvolution` owns the bijection between the two, so a solver
step can be dropped into a `DynamicalModel` without any reshaping in user code.

```python
transition = SpatialStateEvolution(
    stepper=my_stepper,               # field -> field, one solver step
    field_shape=(3, 64, 64, 64),
    dt_obs=0.5,
    n_substeps=10,
)
```

The `stepper` is applied `n_substeps` times per transition under `jax.lax.scan`. Transitions
are deterministic by default -- the returned distribution is a `Delta` at the stepped field.
Passing `process_noise_std` adds white Gaussian noise once per transition, turning it into a
diagonal `Normal`; the matching `observation_noise_std` on
[field_observation][dynestyx.models.spatial.field_observation] and `spatial_dynamics`
switches the observation between exact (Dirac) and Gaussian. Both are held as scale
*vectors*, so nothing of size `state_dim ** 2` is ever built.

**External (non-JAX) solvers.** A stepper may call out to a host-side solver through
`jax.pure_callback`, which works under the `lax.scan` used here. Three caveats:

- The callback receives a concrete JAX array, not a `numpy.ndarray`; convert with
  `np.asarray` before handing it to a library that requires a real NumPy buffer.
- Pass `vmap_method="sequential"` to `jax.pure_callback`. The simulators `vmap` over
  `n_simulations` -- even when it is 1 -- and the default `vmap_method=None` raises.
  `"expand_dims"` and `"broadcast_all"` return *silently wrong* results unless the callback
  genuinely handles a leading batch axis.
- `jax.pure_callback` is not differentiable, so such a model cannot be used with
  `EKFConfig`, or with gradient-based inference over solver parameters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, SupportsInt, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import Array
from jaxtyping import Real
from numpyro.distributions import Distribution

from dynestyx.models.core import DiscreteTimeStateEvolution, DynamicalModel
from dynestyx.models.observations import (
    DiagonalGaussianObservation,
    DiracIdentityObservation,
)
from dynestyx.utils import _raise_now_or_error_if

__all__ = [
    "FieldLayout",
    "SpatialStateEvolution",
    "field_observation",
    "spatial_dynamics",
]

FieldShape = tuple[int, ...]
FieldShapeLike = Sequence[int] | Sequence[Sequence[int]]
Fields = Real[Array, "..."] | tuple[Real[Array, "..."], ...]
NoiseStdLike = float | int | Real[Array, "..."] | Sequence[Any]
NoiseScale = Real[Array, "..."]


def _normalize_field_shape(field_shape: FieldShapeLike) -> tuple[FieldShape, ...]:
    """Coerce the user-facing `field_shape` argument into a tuple of field shapes.

    A single field may be given as a flat sequence of ints, `(3, 64, 64, 64)`; several
    fields as a sequence of such sequences, `((2, 64, 64), (1, 200, 200))`.
    This second case is useful for coupled PDEs, e.g. a velocity field and a scalar tracer, of different spatial resolutions.

    The two are distinguished by whether the first element is itself a sequence.
    """
    if not isinstance(field_shape, Sequence) or len(field_shape) == 0:
        raise ValueError(
            "field_shape must be a non-empty sequence of ints, or a non-empty sequence "
            f"of such sequences; got {field_shape!r}."
        )

    single = not isinstance(field_shape[0], Sequence)
    specs_in: list[Any] = [field_shape] if single else [spec for spec in field_shape]

    specs: list[FieldShape] = []
    for i, spec in enumerate(specs_in):
        if not isinstance(spec, Sequence) or len(spec) == 0:
            raise ValueError(
                f"field_shape[{i}] must be a non-empty sequence of ints; got {spec!r}."
            )
        dims = tuple(int(cast(SupportsInt, d)) for d in spec)
        if any(d <= 0 for d in dims):
            raise ValueError(
                f"field_shape[{i}] must have strictly positive dimensions; got {dims}."
            )
        specs.append(dims)
    return tuple(specs)


def _resolve_noise_scale(
    layout: FieldLayout,
    std: NoiseStdLike | None,
    *,
    name: str,
) -> NoiseScale | None:
    """Broadcast a white-noise standard deviation onto the flat state vector.

    Accepts, in order of increasing specificity:

    - `None` -- no noise.
    - A scalar, kept as a 0-d array so that nothing of size `state_dim` is
      materialised.
    - A sequence with one entry per field, each entry itself a scalar or an array
      broadcastable to that field's shape (so a per-component std of a velocity field is
      `jnp.array([1e-3, 1e-3, 1e-4])[:, None, None, None]`).
    - For a single-field layout only, an array broadcastable to the field shape.

    Returns:
        `None`, a 0-d array, or a `(state_dim,)` vector of standard deviations.
    """
    if std is None:
        return None

    if isinstance(std, Sequence) and not isinstance(std, str):
        if len(std) != layout.n_fields:
            raise ValueError(
                f"{name} given as a sequence must have one entry per field: expected "
                f"{layout.n_fields}, got {len(std)}."
            )
        entries: tuple[Any, ...] = tuple(std)
    else:
        arr = jnp.asarray(std)
        if arr.ndim == 0:
            return _check_positive_scale(arr, name)
        if not layout.is_single:
            raise ValueError(
                f"{name} for a multi-field layout must be a scalar or a sequence of "
                f"{layout.n_fields} per-field entries; got an array of shape "
                f"{arr.shape}."
            )
        entries = (arr,)

    blocks = []
    for i, (entry, shape) in enumerate(zip(entries, layout.shapes, strict=True)):
        arr = jnp.asarray(entry)
        try:
            arr = jnp.broadcast_to(arr, shape)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"{name}[{i}] has shape {arr.shape}, which does not broadcast to field "
                f"shape {shape}."
            ) from exc
        blocks.append(arr.reshape(-1))
    return _check_positive_scale(jnp.concatenate(blocks), name)


def _check_positive_scale(scale: NoiseScale, name: str) -> NoiseScale:
    """Reject non-positive standard deviations; a zero std is a degenerate Normal."""
    return _raise_now_or_error_if(
        scale,
        jnp.any(scale <= 0),
        f"{name} must be strictly positive; pass None for a noise-free (Dirac) model.",
    )


class FieldLayout(eqx.Module):
    """Bijection between named spatial fields and the flat `dynestyx` state vector.

    The flat state is the concatenation of each field's block, in `shapes` order. Within a
    block the layout is C (row-major) order, so a field of shape `(n_components, *spatial)`
    flattens component-first -- the same convention as
    [GridInterpolator][dynestyx.observation.GridInterpolator] and as the native field layout
    of spectral solvers such as `exponax`.

    Attributes:
        shapes: Shape of each field, e.g. `((3, 64, 64, 64),)` for a single 3D vector field.
        sizes: Number of scalars in each field block.
        offsets: Start index of each block within the flat state.
        state_dim: Total flat width; equals `DynamicalModel.state_dim`.

    Note:
        This layout records *shapes only*, not grid coordinates. Observation operators that
        need geometry (interpolation at probe points, staggered-grid offsets) require
        coordinate metadata that this class does not yet carry.
    """

    shapes: tuple[FieldShape, ...] = eqx.field(static=True)
    sizes: tuple[int, ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(self, field_shape: FieldShapeLike):
        """
        Args:
            field_shape: A single field shape `(n_components, *spatial)`, or a sequence of
                such shapes for a coupled multi-field state.
        """
        shapes = _normalize_field_shape(field_shape)
        sizes = tuple(math.prod(shape) for shape in shapes)
        offsets = []
        acc = 0
        for size in sizes:
            offsets.append(acc)
            acc += size
        self.shapes = shapes
        self.sizes = sizes
        self.offsets = tuple(offsets)
        self.state_dim = acc

    @property
    def n_fields(self) -> int:
        """Number of field blocks in the state."""
        return len(self.shapes)

    @property
    def is_single(self) -> bool:
        """Whether the state is one field, so `unflatten` returns a bare array."""
        return len(self.shapes) == 1

    def unflatten(self, x: Real[Array, "*batch state_dim"]) -> Fields:
        """Split a flat state into fields, preserving any leading batch axes.

        Accepts `(state_dim,)`, `(time, state_dim)`, `(n_simulations, time, state_dim)` and
        so on, which is what makes it usable both inside a transition and for reshaping a
        whole trajectory for diagnostics.

        Args:
            x: Flat state with `state_dim` as its trailing axis.

        Returns:
            A single field array when the layout has one field, else a tuple of fields.
            Each field has shape `(*batch, *field_shape)`.
        """
        x = jnp.asarray(x)
        if x.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected trailing axis {self.state_dim}, got array of shape {x.shape}."
            )
        batch = x.shape[:-1]
        fields = tuple(
            jax.lax.dynamic_slice_in_dim(x, offset, size, axis=-1).reshape(
                (*batch, *shape)
            )
            for offset, size, shape in zip(
                self.offsets, self.sizes, self.shapes, strict=True
            )
        )
        return fields[0] if self.is_single else fields

    def flatten(self, fields: Fields) -> Real[Array, "*batch state_dim"]:
        """Concatenate fields back into a flat state, validating each block's shape.

        Args:
            fields: A single field array, or a tuple of field arrays matching `shapes`.

        Returns:
            Flat state of shape `(*batch, state_dim)`.
        """
        field_tuple: tuple[Any, ...] = (
            (fields,) if not isinstance(fields, tuple) else fields
        )
        if len(field_tuple) != self.n_fields:
            raise ValueError(
                f"Expected {self.n_fields} field(s), got {len(field_tuple)}."
            )
        flat = []
        batch: tuple[int, ...] | None = None
        for i, (field, shape) in enumerate(zip(field_tuple, self.shapes, strict=True)):
            arr = jnp.asarray(field)
            ndim = len(shape)
            if arr.ndim < ndim or arr.shape[arr.ndim - ndim :] != shape:
                raise ValueError(
                    f"Field {i} must have trailing shape {shape}; got {arr.shape}."
                )
            field_batch = arr.shape[: arr.ndim - ndim]
            if batch is None:
                batch = field_batch
            elif field_batch != batch:
                raise ValueError(
                    f"Inconsistent batch axes across fields: {batch} vs {field_batch}."
                )
            flat.append(arr.reshape((*field_batch, self.sizes[i])))
        return jnp.concatenate(flat, axis=-1)


class SpatialStateEvolution(DiscreteTimeStateEvolution):
    r"""Discrete-time transition driven by a solver step on spatial fields.

    Wraps a `stepper` -- one step of any solver, JAX-native or external -- into the flat
    state-vector contract `dynestyx` requires:

    $$
    x_{t_{k+1}} = \Phi^{(n)}\!\left(x_{t_k}\right) + \varepsilon_k,
    $$

    where $\Phi^{(n)}$ applies `stepper` `n_substeps` times across the interval, and
    $\varepsilon_k \sim \mathcal{N}(0, \operatorname{diag}(\sigma^2))$ is the optional
    white process noise set by `process_noise_std`. Without it ($\sigma$ unset) the
    transition is deterministic and the returned distribution is a `Delta`; with it, a
    diagonal `Normal`.

    Attributes:
        stepper (Callable): One solver step. Either `fields -> fields`, or
            `(fields, aux) -> (fields, aux)` when `aux_init` is given. `fields` is a single
            array for a one-field layout and a tuple otherwise.
        aux_init (Callable | None): Builds the auxiliary carry from the current fields at
            the start of each transition. See the note on auxiliary quantities below.
        layout (FieldLayout): Field/flat-vector bijection.
        dt_obs (float): The model's time step. Each transition asserts that
            `t_next - t_now` matches this.
        n_substeps (int): Solver steps per transition. Defaults to `1`.
        process_noise_scale (jax.Array | None): Resolved white process-noise standard
            deviation -- `None`, a 0-d array, or a `(state_dim,)` vector. Set from
            `process_noise_std`.

    Note:
        The process noise is **white in space and time** and added **once per transition**,
        after all `n_substeps` solver steps -- not per substep, and not scaled by `dt_obs`.
        `process_noise_std` is therefore the per-transition standard deviation. Grid-white
        noise is unphysical for a PDE (it has no spatial correlation and excites the
        smallest resolved scale hardest) and can destabilise a spectral solver; keep it
        small relative to the field, and prefer spatially correlated noise once available.

    Note:
        The noise is held as a scale *vector*, never a dense `(state_dim, state_dim)`
        covariance -- the shipped `GaussianStateEvolution` materialises one, which is 4.8 TB
        at `state_dim = 786,432`. The returned `Normal(...).to_event(1)` consequently has no
        `covariance_matrix`, so the Kalman backends do not apply; the simulators, the
        ensemble/particle filters and `log_prob` do.

    Note:
        `stepper` is a *dynamic* field, so array-valued solver parameters remain leaves of
        the enclosing `DynamicalModel` pytree. Passing a closure over a module-level solver
        instead hides those arrays from the pytree.

    Note:
        The auxiliary carry is threaded **within** a transition and rebuilt by `aux_init` at
        every transition boundary. It is intended for quantities with no dynamics of their
        own -- a pressure warm start for an iterative Poisson solve, say -- which are
        determined instantaneously by the state. Anything that must genuinely persist across
        transitions is a state variable and belongs in `field_shape`.

    Note:
        `dt_obs` is asserted against `t_next - t_now`, but this class cannot see the
        stepper's own step size. Ensuring `dt_solver * n_substeps == dt_obs` is the caller's
        responsibility.
    """

    # `DynamicalModel.__init__` otherwise validates the state dimension by *executing* the
    # transition on a synthetic unit interval. That is wrong for this class twice over: the
    # probe would run a full solver step on every model construction, and its hardcoded
    # `t_next = t_now + 1.0` trips the `dt_obs` check for any model whose step is not 1.0.
    # The layout already pins the output width exactly, and `__init__` verifies the
    # stepper against it abstractly, so the probe adds nothing.
    _dynestyx_discretizer_preserves_state_shape: ClassVar[bool] = True

    stepper: Callable[..., Any]
    aux_init: Callable[[Any], Any] | None
    process_noise_scale: NoiseScale | None
    layout: FieldLayout = eqx.field(static=True)
    dt_obs: float = eqx.field(static=True)
    n_substeps: int = eqx.field(static=True)

    def __init__(
        self,
        stepper: Callable[..., Any],
        *,
        field_shape: FieldShapeLike,
        dt_obs: float,
        n_substeps: int = 1,
        process_noise_std: NoiseStdLike | None = None,
        aux_init: Callable[[Any], Any] | None = None,
        validate: bool = True,
    ):
        """
        Args:
            stepper: One solver step, `fields -> fields`, or `(fields, aux) -> (fields, aux)`
                when `aux_init` is given.
            field_shape: A single field shape `(n_components, *spatial)`, or a sequence of
                such shapes for a coupled multi-field state.
            dt_obs: The model's time step, spanned by `n_substeps` solver steps.
            n_substeps: Solver steps per transition. Must be at least 1.
            process_noise_std: Standard deviation of the additive white process noise, or
                `None` (the default) for a deterministic `Delta` transition. A scalar
                applies everywhere; a sequence gives one entry per field, each a scalar or
                an array broadcastable to that field's shape.
            aux_init: Optional builder for the auxiliary carry, `fields -> aux`.
            validate: Check at construction that `stepper` returns fields matching
                `field_shape`. Uses `jax.eval_shape`, so no solver work is performed and no
                host callback fires. Set `False` for a stepper that cannot be abstractly
                traced.
        """
        if n_substeps < 1:
            raise ValueError(f"n_substeps must be at least 1; got {n_substeps}.")
        dt_obs = float(dt_obs)
        if not dt_obs > 0.0:
            raise ValueError(f"dt_obs must be strictly positive; got {dt_obs}.")
        self.stepper = stepper
        self.aux_init = aux_init
        self.layout = FieldLayout(field_shape)
        self.dt_obs = dt_obs
        self.n_substeps = n_substeps
        self.process_noise_scale = _resolve_noise_scale(
            self.layout, process_noise_std, name="process_noise_std"
        )
        if validate:
            self._validate_stepper_shapes()

    def _validate_stepper_shapes(self) -> None:
        """Confirm abstractly that `stepper` preserves the declared field shapes."""
        probe = self.layout.unflatten(jnp.zeros(self.layout.state_dim))
        try:
            out, _ = jax.eval_shape(self.rollout, probe)
        except Exception as exc:  # noqa: BLE001 - re-raised with actionable context
            raise ValueError(
                "Could not trace `stepper` to validate its output shape. Check that it "
                "accepts the declared field layout, or pass validate=False."
            ) from exc
        out_tuple = out if isinstance(out, tuple) else (out,)
        if len(out_tuple) != self.layout.n_fields:
            raise ValueError(
                f"stepper returned {len(out_tuple)} field(s), but field_shape declares "
                f"{self.layout.n_fields}."
            )
        for i, (got, expected) in enumerate(
            zip(out_tuple, self.layout.shapes, strict=True)
        ):
            if tuple(got.shape) != expected:
                raise ValueError(
                    f"stepper returned field {i} with shape {tuple(got.shape)}, but "
                    f"field_shape declares {expected}."
                )

    @property
    def state_dim(self) -> int:
        """Flat state width implied by `field_shape`."""
        return self.layout.state_dim

    def rollout(self, fields: Fields) -> tuple[Fields, Any]:
        """Apply `stepper` `n_substeps` times, returning the fields and the final carry.

        Exposed so callers can inspect the auxiliary carry, which the transition itself
        discards.

        Args:
            fields: Starting fields, matching the layout.

        Returns:
            `(fields, aux)` after `n_substeps` steps. `aux` is `None` when `aux_init` is not
            set.
        """
        if self.aux_init is None:
            if self.n_substeps == 1:
                return self.stepper(fields), None

            def body_no_aux(carry, _):
                return self.stepper(carry), None

            out, _ = jax.lax.scan(body_no_aux, fields, None, length=self.n_substeps)
            return out, None

        aux = self.aux_init(fields)
        if self.n_substeps == 1:
            return self.stepper(fields, aux)

        def body_aux(carry, _):
            return self.stepper(*carry), None

        (out, aux_out), _ = jax.lax.scan(
            body_aux, (fields, aux), None, length=self.n_substeps
        )
        return out, aux_out

    def __call__(
        self,
        x: Real[Array, " state_dim"],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t_now: float | int | Real[Array, ""],
        t_next: float | int | Real[Array, ""],
    ) -> Distribution:
        interval = jnp.asarray(t_next) - jnp.asarray(t_now)
        tolerance = 1e-8 * max(self.dt_obs, 1.0)
        x = _raise_now_or_error_if(
            jnp.asarray(x),
            jnp.abs(interval - self.dt_obs) > tolerance,
            f"SpatialStateEvolution expects transitions of length dt_obs={self.dt_obs}, "
            "but t_next - t_now differs. Align predict_times/obs_times with dt_obs.",
        )
        fields, _ = self.rollout(self.layout.unflatten(x))
        loc = self.layout.flatten(fields)
        if self.process_noise_scale is None:
            return dist.Delta(loc, event_dim=1)
        return dist.Normal(loc, self.process_noise_scale).to_event(1)


def field_observation(
    *,
    field_shape: FieldShapeLike,
    noise_std: NoiseStdLike | None = None,
) -> DiracIdentityObservation | DiagonalGaussianObservation:
    r"""Build the observation model that observes every field point directly.

    One entry point for the two noise-free/noisy variants, so switching between them is a
    single argument rather than a different class:

    - `noise_std=None` (default) gives a
      [DiracIdentityObservation][dynestyx.models.observations.DiracIdentityObservation],
      $y_t = x_t$ exactly.
    - Any other value gives a
      [DiagonalGaussianObservation][dynestyx.models.observations.DiagonalGaussianObservation],
      $y_t \sim \mathcal{N}(x_t, \operatorname{diag}(\sigma^2))$, with white noise.

    Args:
        field_shape: A single field shape, or a sequence of shapes for a coupled state.
            Used only to broadcast a per-field `noise_std`.
        noise_std: Observation noise standard deviation, broadcast exactly as
            `SpatialStateEvolution.process_noise_std` is: a scalar everywhere, or one
            entry per field.

    Returns:
        An observation model with `observation_dim == state_dim`.

    Note:
        Observing the full field means the simulator stores a second copy of every state --
        2x memory, exactly. For large fields, observe a subset instead.
    """
    scale = _resolve_noise_scale(FieldLayout(field_shape), noise_std, name="noise_std")
    if scale is None:
        return DiracIdentityObservation()
    return DiagonalGaussianObservation(scale)


def spatial_dynamics(
    stepper: Callable[..., Any],
    *,
    field_shape: FieldShapeLike,
    dt_obs: float,
    initial_field: Fields,
    n_substeps: int = 1,
    initial_std: NoiseStdLike | None = None,
    process_noise_std: NoiseStdLike | None = None,
    observation_noise_std: NoiseStdLike | None = None,
    aux_init: Callable[[Any], Any] | None = None,
) -> DynamicalModel:
    """Build a `DynamicalModel` from a solver step and an initial field.

    A convenience wrapper composing a `Delta` initial condition, a
    [SpatialStateEvolution][dynestyx.models.spatial.SpatialStateEvolution], and a noise-free
    identity observation model.

    Args:
        stepper: One solver step. See `SpatialStateEvolution`.
        field_shape: A single field shape, or a sequence of shapes for a coupled state.
        dt_obs: The model's time step.
        initial_field: Initial fields, matching `field_shape`.
        n_substeps: Solver steps per transition.
        initial_std: Standard deviation of the Gaussian spread around `initial_field`, or
            `None` for a `Delta` (exactly known) initial condition. Broadcast like
            `process_noise_std`.
        process_noise_std: Standard deviation of the white process noise, or `None` for a
            deterministic transition. See `SpatialStateEvolution`.
        observation_noise_std: Standard deviation of the white observation noise, or `None`
            for exact (Dirac) observation of the state.
        aux_init: Optional auxiliary-carry builder.

    Returns:
        DynamicalModel: An uncontrolled, fully observed spatial model, deterministic and
        noise-free unless `initial_std` / `process_noise_std` / `observation_noise_std`
        say otherwise.

    Note:
        A `Delta` initial condition has a degenerate log density, which makes the
        linearising filters (`EKFConfig`) return `NaN` even when the transition and
        observation are Gaussian. Set `initial_std` when filtering.

    Note:
        The observation model comes from
        [field_observation][dynestyx.models.spatial.field_observation], so
        `observation_dim == state_dim` and the simulator stores a full copy of every
        state. For large fields, supply a subsampling observation model instead by building
        the `DynamicalModel` directly.
    """
    state_evolution = SpatialStateEvolution(
        stepper,
        field_shape=field_shape,
        dt_obs=dt_obs,
        n_substeps=n_substeps,
        process_noise_std=process_noise_std,
        aux_init=aux_init,
    )
    initial_flat = state_evolution.layout.flatten(initial_field)
    initial_scale = _resolve_noise_scale(
        state_evolution.layout, initial_std, name="initial_std"
    )
    initial_condition = (
        dist.Delta(initial_flat, event_dim=1)
        if initial_scale is None
        else dist.Normal(initial_flat, initial_scale).to_event(1)
    )
    return DynamicalModel(
        initial_condition=initial_condition,
        state_evolution=state_evolution,
        observation_model=field_observation(
            field_shape=field_shape, noise_std=observation_noise_std
        ),
        control_dim=0,
    )
