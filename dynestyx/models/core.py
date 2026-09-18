"""Core interfaces and base classes for dynamical models."""

from __future__ import annotations

from copy import copy
from typing import Any, Protocol, cast, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Real
from numpyro.distributions import Distribution

from dynestyx.models.checkers import (
    _infer_observation_dim_in_plate_context,
    _infer_vector_dim_from_distribution,
    _inside_numpyro_plate_context,
    _is_categorical_distribution,
    _make_probe_state,
    _validate_categorical_state,
    _validate_continuous_state_evolution,
    _validate_continuous_time_flag,
    _validate_discrete_state_evolution_output_shape,
    _validate_imex_potential_conflict,
    _validate_observation_dim,
    _validate_state_dim,
)
from dynestyx.models.diffusions import Diffusion
from dynestyx.models.drifts import Drift, Potential
from dynestyx.models.layout import Layout
from dynestyx.types import as_scalar_time_array


@runtime_checkable
class DiscreteStateTransition(Protocol):
    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t_now: float | int | Real[Array, ""],
        t_next: float | int | Real[Array, ""],
    ) -> Distribution:
        raise NotImplementedError()


@runtime_checkable
class ObservationCallable(Protocol):
    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Distribution:
        raise NotImplementedError()


type StateEvolutionLike = (
    ContinuousTimeStateEvolution | DiscreteTimeStateEvolution | DiscreteStateTransition
)
type ObservationModelLike = ObservationModel | ObservationCallable


class DynamicalModel(eqx.Module):
    """
    Unified interface for state-space dynamical systems.

    A dynamical model specifies the joint generative process for states and observations.
    The state evolves according to either a continuous-time SDE or a discrete-time Markov
    transition, and observations are emitted conditionally on the latent state:

    $$
    \\begin{aligned}
    x_0 &\\sim p(x_0) \\\\
    x_{t+1} &\\sim p(x_{t+1} \\mid x_t, u_t, t) \\\\
    y_t &\\sim p(y_t \\mid x_t, u_t, t)
    \\end{aligned}
    $$

    For continuous-time models, the state evolution is governed by an SDE (see
    `ContinuousTimeStateEvolution`). For discrete-time models, the transition
    is given by `DiscreteTimeStateEvolution`.

    Attributes:
        state_dim (int): Dimension of the latent state vector $x_t \\in \\mathbb{R}^{d_x}$.
        observation_dim (int): Dimension of the observation vector $y_t \\in \\mathbb{R}^{d_y}$.
        categorical_state (bool): Whether latent states are categorical class labels.
            Gets inferred automatically from the type of `initial_condition`.
        control_dim (int): Dimension of the control/input vector $u_t \\in \\mathbb{R}^{d_u}$. Defaults to 0 if not provided (assumes no controls).
        initial_condition (numpyro.distributions.Distribution): Distribution over the initial state $p(x_0)$.
            Pass a NumPyro distribution instance (i.e., a `numpyro.distributions.Distribution` subclass). See the
            [NumPyro distributions API](https://num.pyro.ai/en/stable/distributions.html).
        state_layout (Layout | None): Layout for callable state inputs and state results. Defaults
            to None; not inferred from the transition or observation model.
        control_layout (Layout | None): Layout for callable control inputs.
            Infers control_dim when provided; external control arrays remain flat.
        observation_layout (Layout | None): Layout for observation results and
            helper output flattening. Custom callables return flat-event distributions.
        state_evolution (ContinuousTimeStateEvolution | DiscreteTimeStateEvolution | Callable): The state transition model.
            Use `ContinuousTimeStateEvolution` for SDEs or `DiscreteTimeStateEvolution` for discrete-time Markov
            transitions. A callable is also accepted (e.g., `lambda x, u, t_now, t_next: ...`), but class-based
            implementations are recommended for full compatibility with type-based integrations (such as automatic
            simulator selection).
        observation_model (ObservationModel | Callable): The observation/likelihood model $p(y_t \\mid x_t, u_t, t)$.
            A callable is accepted (e.g., `lambda x, u, t: ...`) as long as it returns a NumPyro-compatible
            distribution, while subclassing `ObservationModel` is recommended for richer reuse and consistency.
        control_model (Any): Optional model for control inputs (e.g., exogenous process). Not currently supported.
        t0 (float | Array | None): Optional declared start time of the model. If ``None`` (default), the start time
            is auto-inferred as ``obs_times[0]`` when the simulator runs and recorded as a
            ``numpyro.deterministic("t0", ...)`` site. If provided, it must match ``obs_times[0]``
            exactly; a mismatch raises a ``ValueError`` at simulation time.
        continuous_time (bool): Whether the model uses continuous-time state evolution (SDE) or discrete-time.
            Gets set automatically from the concrete type of `state_evolution`.
    
    Note:
        - `continuous_time`, `state_dim`, `observation_dim`, and `categorical_state` are inferred automatically; do not pass them to the constructor.
        - Logic for control_model is not implemented yet.
        - `t0` different from `obs_times[0]` is not supported yet.

    Plate note:
        Inside ``dsx.plate``, state and observation dimensions are inferred from
        distribution ``event_shape`` rather than leading plate batch axes. This
        allows batched initial-condition parameters such as
        ``initial_mean.shape == (N, state_dim)``.
    
    """

    initial_condition: Distribution
    state_evolution: StateEvolutionLike
    observation_model: ObservationModelLike
    control_dim: int
    control_model: Any
    t0: Real[Array, ""] | None
    state_dim: int
    observation_dim: int
    categorical_state: bool
    continuous_time: bool
    state_layout: Layout | None = eqx.field(static=True, default=None, kw_only=True)
    control_layout: Layout | None = eqx.field(static=True, default=None, kw_only=True)
    observation_layout: Layout | None = eqx.field(
        static=True, default=None, kw_only=True
    )

    def __init__(
        self,
        initial_condition: Distribution,
        state_evolution: StateEvolutionLike,
        observation_model: ObservationModelLike,
        control_dim: int | None = None,
        control_model=None,
        *,
        t0: float | int | Array | None = None,
        state_dim: int | None = None,
        observation_dim: int | None = None,
        categorical_state: bool | None = None,
        continuous_time: bool | None = None,
        state_layout: Layout | None = None,
        control_layout: Layout | None = None,
        observation_layout: Layout | None = None,
    ):
        inferred_continuous_time = isinstance(
            state_evolution, ContinuousTimeStateEvolution
        )
        _validate_continuous_time_flag(continuous_time, inferred_continuous_time)
        self.continuous_time = inferred_continuous_time
        self.state_layout = state_layout
        self.control_layout = control_layout
        self.observation_layout = observation_layout
        # Layouts are static Equinox metadata, not replaceable pytree leaves.
        # Bind missing helper output layouts on shallow copies; arrays are shared
        # and the caller's original components remain untouched.
        if (
            isinstance(state_evolution, DiscreteTimeStateEvolution)
            and state_layout is not None
            and state_evolution.state_layout is None
        ):
            state_evolution = copy(state_evolution)
            object.__setattr__(state_evolution, "state_layout", state_layout)
        if isinstance(observation_model, ObservationModel):
            updates = {
                name: layout
                for name, layout in (
                    ("state_layout", state_layout),
                    ("observation_layout", observation_layout),
                )
                if layout is not None and getattr(observation_model, name) is None
            }
            if updates:
                observation_model = copy(observation_model)
                for name, layout in updates.items():
                    object.__setattr__(observation_model, name, layout)
        self.initial_condition = initial_condition
        self.state_evolution = state_evolution
        self.observation_model = observation_model
        self.control_model = control_model
        self.t0 = None if t0 is None else as_scalar_time_array(t0, name="t0")

        # State-dim inference depends on whether we're inside a plate.
        #
        # Outside plates, ``_infer_vector_dim_from_distribution`` reads the
        # distribution's full sample shape: a rank-1 sample is treated as a
        # vector event of that length. This is the historical convention and
        # matches typical single-trajectory usage.
        #
        # Inside a plate, that convention is wrong. A sample of shape ``(N,
        # d)`` is plate-batched (``N`` independent members, each with a
        # ``d``-dim state); reading the full sample shape would misinterpret
        # ``N`` as the state dim. We switch to ``event_shape``-based inference
        # via ``allow_batch_shape=True`` so leading plate axes are ignored.
        #
        # See ``dsx.plate`` in dynestyx/handlers.py for the user-facing shape
        # contract this enforces.
        _inside_plate = _inside_numpyro_plate_context()

        inferred_state_dim = _infer_vector_dim_from_distribution(
            initial_condition,
            "initial_condition",
            allow_batch_shape=_inside_plate,
        )
        _validate_state_dim(state_dim, inferred_state_dim)
        layouts = [
            getattr(component, "state_layout", None)
            for component in (state_evolution, observation_model)
        ]
        layouts = [layout for layout in [state_layout, *layouts] if layout is not None]
        for layout in layouts:
            if layout is not None and layout.state_dim != inferred_state_dim:
                raise ValueError(
                    "Layout dimension must match initial_condition state dimension."
                )
        if layouts and any(layout != layouts[0] for layout in layouts[1:]):
            raise ValueError(
                "Model, transition, and observation state layouts must agree."
            )
        inferred_categorical_state = _is_categorical_distribution(initial_condition)
        _validate_categorical_state(categorical_state, inferred_categorical_state)
        if control_layout is not None:
            if control_dim is not None and control_dim != control_layout.state_dim:
                raise ValueError("control_dim must match control_layout dimension.")
            control_dim = control_layout.state_dim
        elif control_dim is None:
            control_dim = 0

        # Skip shape validation when inside a numpyro plate context, since
        # batched parameters produce shapes that don't match unbatched expectations.

        def _make_probes() -> tuple[
            Real[Array, " state_dim"] | Real[Array, ""],
            Real[Array, " control_dim"] | Real[Array, ""] | None,
            Real[Array, ""],
        ]:
            """Build synthetic inputs for validation/metadata resolution."""
            x_probe = _make_probe_state(
                initial_condition=initial_condition,
                state_dim=inferred_state_dim,
            )
            u_probe = None if control_dim == 0 else jnp.zeros((control_dim,))
            t_probe = jnp.array(0.0) if self.t0 is None else self.t0
            return x_probe, u_probe, t_probe

        x_probe, u_probe, t_probe = _make_probes()

        def _resolve_continuous_state_evolution(
            current_state_evolution: ContinuousTimeStateEvolution,
        ) -> (
            DeterministicContinuousTimeStateEvolution
            | StochasticContinuousTimeStateEvolution
        ):
            """Return either a DeterministicContinuousTimeStateEvolution or a StochasticContinuousTimeStateEvolution.
            If diffusion is present, lazily build probes to resolve its metadata (e.g., bm_dim).
            """
            diffusion = current_state_evolution.diffusion
            if diffusion is None:
                if isinstance(
                    current_state_evolution, DeterministicContinuousTimeStateEvolution
                ):
                    return current_state_evolution
                return DeterministicContinuousTimeStateEvolution(
                    drift=current_state_evolution.drift,
                    potential=current_state_evolution.potential,
                    use_negative_gradient=current_state_evolution.use_negative_gradient,
                )

            resolved_diffusion = diffusion.resolve_metadata(
                state_dim=inferred_state_dim,
                x_probe=x_probe,
                u_probe=u_probe,
                t_probe=t_probe,
            )
            return StochasticContinuousTimeStateEvolution(
                drift=current_state_evolution.drift,
                potential=current_state_evolution.potential,
                use_negative_gradient=current_state_evolution.use_negative_gradient,
                diffusion=resolved_diffusion,
            )

        if self.continuous_time:
            # Needs no shape/probe data, so unlike the checks below it must
            # run unconditionally -- including inside dsx.plate, where those
            # shape checks are skipped.
            _validate_imex_potential_conflict(state_evolution)

        if _inside_plate:
            # Cannot validate shapes with batched parameters; trust the user.
            # Infer observation_dim from observation model if not explicitly provided.
            inferred_obs_dim = _infer_observation_dim_in_plate_context(
                observation_model=self.observation_distribution,
                x_probe=x_probe,
                u_probe=u_probe,
                t_probe=t_probe,
                observation_dim=observation_dim,
            )
        else:
            if self.continuous_time:
                _validate_continuous_state_evolution(
                    state_evolution=state_evolution,
                    state_dim=inferred_state_dim,
                    x_probe=x_probe,
                    u_probe=u_probe,
                    t_probe=t_probe,
                )
            else:
                structured_x, structured_u = self._structured_inputs(x_probe, u_probe)
                _validate_discrete_state_evolution_output_shape(
                    state_evolution=state_evolution,
                    state_dim=inferred_state_dim,
                    x_probe=structured_x,
                    u_probe=structured_u,
                    t_probe=t_probe,
                )

            obs_dist = self.observation_distribution(x_probe, u_probe, t_probe)
            inferred_obs_dim = _infer_vector_dim_from_distribution(
                obs_dist, "observation_model(x, u, t)"
            )
            _validate_observation_dim(observation_dim, inferred_obs_dim)

        if self.continuous_time:
            continuous_state_evolution = cast(
                ContinuousTimeStateEvolution, state_evolution
            )
            self.state_evolution = _resolve_continuous_state_evolution(
                continuous_state_evolution,
            )

        component_observation_layout = getattr(
            observation_model, "observation_layout", None
        )
        if (
            observation_layout is not None
            and component_observation_layout is not None
            and observation_layout != component_observation_layout
        ):
            raise ValueError("Model and observation output layouts must agree.")
        validation_observation_layout = (
            observation_layout or component_observation_layout
        )
        if (
            validation_observation_layout is not None
            and validation_observation_layout.state_dim != inferred_obs_dim
        ):
            raise ValueError(
                "Observation layout dimension must match the observation distribution."
            )

        self.state_dim = int(inferred_state_dim)
        # Scalar observation variance needs the output width before it can
        # be stored as a dense covariance for all inference backends.
        from dynestyx.models.observations import GaussianObservation
        from dynestyx.models.state_evolution import GaussianStateEvolution

        if isinstance(self.state_evolution, GaussianStateEvolution):
            self.state_evolution = self.state_evolution.resolve_covariance(
                inferred_state_dim
            )
        if isinstance(self.observation_model, GaussianObservation):
            self.observation_model = self.observation_model.resolve_covariance(
                inferred_obs_dim
            )
        self.observation_dim = int(inferred_obs_dim)
        self.control_dim = int(control_dim)
        self.categorical_state = bool(inferred_categorical_state)

    def _structured_inputs(self, x, u):
        if self.state_layout is not None:
            x = self.state_layout.unflatten(x)
        if self.control_layout is not None and u is not None:
            u = self.control_layout.unflatten(u)
        return x, u

    def transition_distribution(self, x, u, t_now, t_next):
        """Evaluate the transition on structured inputs; return a flat distribution."""
        x, u = self._structured_inputs(x, u)
        return self.state_evolution(x, u, t_now, t_next)

    def observation_distribution(self, x, u, t):
        """Evaluate observations on structured inputs; return a flat distribution."""
        x, u = self._structured_inputs(x, u)
        return self.observation_model(x, u, t)

    def transition_mean(self, x, u, t_now, t_next):
        """Flat Gaussian transition mean for inference backends."""
        x, u = self._structured_inputs(x, u)
        mean = getattr(self.state_evolution, "mean", None)
        if mean is not None:
            return mean(x, u, t_now, t_next)
        return self.state_evolution(x, u, t_now, t_next).mean

    def observation_mean(self, x, u, t):
        """Flat Gaussian observation mean for inference backends."""
        x, u = self._structured_inputs(x, u)
        mean = getattr(self.observation_model, "mean", None)
        if mean is not None:
            return mean(x, u, t)
        return self.observation_model(x, u, t).mean


class ContinuousTimeStateEvolution(eqx.Module):
    """
    Continuous-time state evolution via stochastic differential equations (SDEs).

    The state evolves according to

    $$
    dx_t = \\bigl[ f(x_t, u_t, t) + s \\, \\nabla_x V(x_t, u_t, t) \\bigr] \\, dt
         + L(x_t, u_t, t) \\, dW_t
    $$

    where $f$ is the drift, $V$ is an optional potential, and $L$ is the diffusion
    coefficient. The sign $s$ is $-1$ when `use_negative_gradient` is True (e.g., for
    Langevin dynamics) and $+1$ otherwise.

    Attributes:
        drift (Drift | None): Drift vector field $f(x, u, t)$.
            Defaults to zero if None.
            At least one of `drift` or `potential` must be non-None.
        potential (Potential | None): Scalar potential $V(x, u, t)$ whose gradient is added to the drift.
            Defaults to zero if None.
            At least one of `drift` or `potential` must be non-None.
        use_negative_gradient (bool): If True, use $-\\nabla_x V$ (e.g., gradient descent on potential);
            otherwise use $+\\nabla_x V$. Default is False.
        diffusion (Diffusion | None): Diffusion coefficient object.
            Use `FullDiffusion`, `DiagonalDiffusion`, or `ScalarDiffusion` to define
            the stochastic part of the SDE. Pass `None` for deterministic dynamics.

    Note:
        For IMEX (implicit-explicit) diffrax solvers (e.g. `diffrax.KenCarp3/4/5`,
        `diffrax.Sil3`), pass an `ImExDrift` instance as `drift` instead of a
        plain callable; see `dynestyx.models.drifts.ImExDrift`.
    """

    drift: Drift | None = None
    potential: Potential | None = None
    use_negative_gradient: bool = eqx.field(static=True, default=False)
    diffusion: Diffusion | None = None

    def total_drift(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Real[Array, " state_dim"] | Real[Array, ""]:
        base = self.drift(x, u, t) if self.drift is not None else None

        potential = self.potential
        if potential is None:
            if base is None:
                raise ValueError(
                    "ContinuousTimeStateEvolution requires drift or potential to be defined."
                )
            return base

        grad_potential = jax.grad(lambda z: potential(z, u, t))(x)
        sign = -1.0 if self.use_negative_gradient else 1.0
        grad_term = sign * grad_potential
        if base is None:
            return grad_term
        return base + grad_term


class DeterministicContinuousTimeStateEvolution(ContinuousTimeStateEvolution):
    """Continuous-time state evolution with no diffusion term, i.e., describing an ODE.

    This is a refined form of :class:`ContinuousTimeStateEvolution` used when a
    model has deterministic continuous-time dynamics, i.e. an ODE rather than an
    SDE. In most user code you should construct
    :class:`ContinuousTimeStateEvolution` directly and let
    :class:`DynamicalModel` refine it into this subclass when ``diffusion=None``.
    Its main semantic guarantee is that ``diffusion`` is always ``None``.
    """

    diffusion: None = eqx.field(static=True, default=None)

    def __init__(
        self,
        drift: Drift | None = None,
        potential: Potential | None = None,
        use_negative_gradient: bool = False,
        diffusion: None = None,
    ):
        if diffusion is not None:
            raise ValueError(
                "DeterministicContinuousTimeStateEvolution does not accept diffusion."
            )
        self.drift = drift
        self.potential = potential
        self.use_negative_gradient = use_negative_gradient
        self.diffusion = None


class StochasticContinuousTimeStateEvolution(ContinuousTimeStateEvolution):
    """Continuous-time state evolution with resolved stochastic diffusion.

    This is a refined form of :class:`ContinuousTimeStateEvolution` used for SDE
    models after the diffusion metadata has been resolved. In practice that means
    the attached :class:`~dynestyx.models.diffusions.Diffusion` has a known
    ``bm_dim`` and can therefore be used safely by downstream SDE solvers,
    discretizers, and inference backends.
    """

    diffusion: Diffusion = eqx.field(kw_only=True)

    def __init__(
        self,
        *,
        drift: Drift | None = None,
        potential: Potential | None = None,
        use_negative_gradient: bool = False,
        diffusion: Diffusion,
    ):
        if diffusion.bm_dim is None:
            raise ValueError(
                "StochasticContinuousTimeStateEvolution requires diffusion with "
                "resolved bm_dim."
            )
        self.drift = drift
        self.potential = potential
        self.use_negative_gradient = use_negative_gradient
        self.diffusion = diffusion

    @property
    def bm_dim(self) -> int:
        bm_dim = self.diffusion.bm_dim
        assert bm_dim is not None
        return bm_dim


class DiscreteTimeStateEvolution(eqx.Module):
    """
    Discrete-time state evolution via Markov transition distributions.

    The next state is drawn from a conditional distribution given the current state,
    control, and time indices:

    $$
    x_{t_{k+1}} \\sim p\\left(x_{t_{k+1}} \\mid x_{t_k}, u_{t_k}, t_k, t_{k+1}\\right)
    $$

    Implementations must return a NumPyro-compatible distribution (e.g.,
    `numpyro.distributions.Distribution`). Most transitions provide sampling,
    moments, and `log_prob`; explicitly sample-only transitions are also valid
    for simulators, ensemble filters, and bootstrap particle filters that never
    evaluate the transition density. Such transitions should raise a targeted
    error when an unavailable moment or density is requested.

    Args:
        x (State): Current state, structured when the model declares a state layout.
        u (Control | None): Current control, structured when a control layout is declared, or None.
        t_now (Time): Current time index $t_k$.
        t_next (Time): Next time index $t_{k+1}$ (for non-uniform sampling or continuous-time embeddings).
        state_layout (Layout | None): Optional structured state metadata.

    Returns:
        numpyro.distributions.Distribution: Distribution over the next state $x_{t_{k+1}}$.
            In practice this should be a `numpyro.distributions.Distribution` instance.
    """

    state_layout: Layout | None = eqx.field(static=True, default=None, kw_only=True)

    def _flatten_state(self, x):
        return x if self.state_layout is None else self.state_layout.flatten(x)

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t_now: float | int | Real[Array, ""],
        t_next: float | int | Real[Array, ""],
    ) -> Distribution:
        raise NotImplementedError()


class ObservationModel(eqx.Module):
    """
    Observation or emission model for state-space systems.

    Defines the conditional distribution of observations given the latent state,
    control, and time:

    $$
    y_t \\sim p(y_t \\mid x_t, u_t, t)
    $$

    Subclasses implement `__call__` to return a NumPyro-compatible distribution.
    The base class provides `log_prob` and `sample` for convenience. Subclasses
    may add parameters (e.g., observation noise scale) as module attributes.

    Methods:
        __call__(x, u, t) -> numpyro.distributions.Distribution: Return the observation distribution (a NumPyro distribution; see
            the [NumPyro distributions API](https://num.pyro.ai/en/stable/distributions.html)) for
            $p(y_t \\mid x_t, u_t, t)$.
        log_prob(y_t, x_t, u_t, t, ...): Compute $\\log p(y_t \\mid x_t, u_t, t)$.
        sample(x, u, t, ...): Sample $y_t \\sim p(y_t \\mid x_t, u_t, t)$.
    """

    state_layout: Layout | None = eqx.field(static=True, default=None, kw_only=True)

    def _flatten_state(self, x):
        return x if self.state_layout is None else self.state_layout.flatten(x)

    observation_layout: Layout | None = eqx.field(
        static=True, default=None, kw_only=True
    )

    def _flatten_observation(self, y):
        return (
            y if self.observation_layout is None else self.observation_layout.flatten(y)
        )

    def __call__(
        self,
        x: Real[Array, " state_dim"] | Real[Array, ""],
        u: Real[Array, " control_dim"] | Real[Array, ""] | None,
        t: float | int | Real[Array, ""],
    ) -> Distribution:
        raise NotImplementedError()

    def log_prob(self, y, x=None, u=None, t=None, *args, **kwargs):
        dist = self(x, u, t)  # type: ignore
        return dist.log_prob(y)

    def sample(self, x, u, t, *args, **kwargs):
        dist = self(x, u, t)
        if "seed" in kwargs:  # for CD-Dynamax compatibility
            seed = kwargs.pop("seed")
            kwargs["key"] = seed
        return dist.sample(*args, **kwargs)
