"""Core interfaces and base classes for dynamical models."""

import dataclasses
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from numpyro._typing import DistributionT

from dynestyx.models.checkers import (
    _infer_vector_dim_from_distribution,
    _is_categorical_distribution,
    _make_probe_state,
    _validate_state_evolution_output_shape,
)
from dynestyx.types import Control, State, Time, dState


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
            In the codebase this is annotated as `DistributionT` (a typing alias); in practice you should pass
            a NumPyro distribution instance (i.e., a `numpyro.distributions.Distribution` subclass). See the
            [NumPyro distributions API](https://num.pyro.ai/en/stable/distributions.html).
        state_evolution (ContinuousTimeStateEvolution | DiscreteTimeStateEvolution | Callable): The state transition model.
            Use `ContinuousTimeStateEvolution` for SDEs or `DiscreteTimeStateEvolution` for discrete-time Markov
            transitions. A callable is also accepted (e.g., `lambda x, u, t_now, t_next: ...`), but class-based
            implementations are recommended for full compatibility with type-based integrations (such as automatic
            simulator selection).
        observation_model (ObservationModel | Callable): The observation/likelihood model $p(y_t \\mid x_t, u_t, t)$.
            A callable is accepted (e.g., `lambda x, u, t: ...`) as long as it returns a NumPyro-compatible
            distribution, while subclassing `ObservationModel` is recommended for richer reuse and consistency.
        control_model (Any): Optional model for control inputs (e.g., exogenous process). Not currently supported.
        t0 (float | None): Optional declared start time of the model. If ``None`` (default), the start time
            is auto-inferred as ``obs_times[0]`` when the simulator runs and recorded as a
            ``numpyro.deterministic("t0", ...)`` site. If provided, it must match ``obs_times[0]``
            exactly; a mismatch raises a ``ValueError`` at simulation time.
        continuous_time (bool): Whether the model uses continuous-time state evolution (SDE) or discrete-time.
            Gets set automatically from the concrete type of `state_evolution`.
    
    Note:
        - `continuous_time`, `state_dim`, `observation_dim`, and `categorical_state` are inferred automatically; do not pass them to the constructor.
        - Logic for control_model is not implemented yet.
        - `t0` different from `obs_times[0]` is not supported yet.
    
    """

    initial_condition: DistributionT
    state_evolution: (
        Callable[[State, Control, Time], State]
        | Callable[[State, Control, Time, Time], State]
    )
    observation_model: Callable[[State, Control, Time], DistributionT]
    control_dim: int
    control_model: Any
    t0: float | None
    state_dim: int
    observation_dim: int
    categorical_state: bool
    continuous_time: bool

    def __init__(
        self,
        initial_condition,
        state_evolution,
        observation_model,
        control_dim: int | None = None,
        control_model=None,
        *,
        t0: float | None = None,
        state_dim: int | None = None,
        observation_dim: int | None = None,
        categorical_state: bool | None = None,
        continuous_time: bool | None = None,
    ):
        inferred_continuous_time = isinstance(
            state_evolution, ContinuousTimeStateEvolution
        )
        if (
            continuous_time is not None
            and bool(continuous_time) != inferred_continuous_time
        ):
            raise ValueError(
                "continuous_time does not match inferred state_evolution type."
            )
        self.continuous_time = inferred_continuous_time
        self.initial_condition = initial_condition
        self.state_evolution = state_evolution
        self.observation_model = observation_model
        self.control_model = control_model
        self.t0 = t0

        inferred_state_dim = _infer_vector_dim_from_distribution(
            initial_condition, "initial_condition"
        )
        if state_dim is not None and int(state_dim) != int(inferred_state_dim):
            raise ValueError(
                "state_dim does not match inferred initial_condition shape. "
                f"Got state_dim={state_dim}, inferred={inferred_state_dim}."
            )
        inferred_categorical_state = _is_categorical_distribution(initial_condition)
        if (
            categorical_state is not None
            and bool(categorical_state) != inferred_categorical_state
        ):
            raise ValueError(
                "categorical_state does not match inferred initial_condition type. "
                f"Got categorical_state={categorical_state}, "
                f"inferred={inferred_categorical_state}."
            )
        if control_dim is None:
            control_dim = 0

        x0 = _make_probe_state(
            initial_condition=initial_condition, state_dim=inferred_state_dim
        )
        u0 = None if control_dim == 0 else jnp.zeros((control_dim,))
        dummy_t0 = jnp.array(0.0) if t0 is None else jnp.array(t0)

        _validate_state_evolution_output_shape(
            state_evolution=state_evolution,
            state_dim=inferred_state_dim,
            x0=x0,
            u0=u0,
            t0=dummy_t0,
            continuous_time=self.continuous_time,
        )

        obs_dist = observation_model(x0, u0, dummy_t0)
        inferred_observation_dim = _infer_vector_dim_from_distribution(
            obs_dist, "observation_model(x, u, t)"
        )
        if observation_dim is not None and int(observation_dim) != int(
            inferred_observation_dim
        ):
            raise ValueError(
                "observation_dim does not match inferred observation_model output shape. "
                f"Got observation_dim={observation_dim}, inferred={inferred_observation_dim}."
            )

        self.state_dim = int(inferred_state_dim)
        self.observation_dim = int(inferred_observation_dim)
        self.control_dim = int(control_dim)
        self.categorical_state = bool(inferred_categorical_state)


class Drift(Protocol):
    """
    Drift vector field for continuous-time state evolution.

    Mathematically, the drift is a mapping
    $\\mu: \\mathbb{R}^{d_x} \\times \\mathbb{R}^{d_u} \\times \\mathbb{R}
    \\to \\mathbb{R}^{d_x}$, i.e., $(x, u, t) \\mapsto \\mu(x, u, t)$.
    In the SDE formulation used by `ContinuousTimeStateEvolution`,
    $dx_t = \\mu(x_t, u_t, t) \\, dt + \\sigma(x_t, u_t, t) \\, dW_t$, this
    mapping forms the $\\mu$ term.

    Implementations should be compatible with JAX transformations (e.g., `jax.jit`,
    `jax.vmap`, and `jax.grad` when differentiable).

    Args:
        x (State): Current state $x \\in \\mathbb{R}^{d_x}$.
        u (Control | None): Current control input $u \\in \\mathbb{R}^{d_u}$ or None.
        t (Time): Current time (scalar or array).

    Returns:
        dState: Drift vector $\\mu(x, u, t) \\in \\mathbb{R}^{d_x}$.

    Note:
        This is a protocol interface; implement this callable signature; do not instantiate.
        We recommend simply using a plain Python function that matches this signature, e.g.:

        ```python
        def drift(x, u, t):
            return - x + u
        ```
        or `lambda x, u, t: - x + u`
    """

    def __call__(
        self,
        x: State,
        u: Control | None,
        t: Time,
    ) -> dState:
        raise NotImplementedError()


class Potential(Protocol):
    """
    Scalar potential energy for gradient-based drift.

    A potential $V(x, u, t)$ maps state, control, and time to a scalar. Its
    gradient contributes to the drift via $\\pm \\nabla_x V(x, u, t)$, enabling
    Langevin-type dynamics. It is used in `ContinuousTimeStateEvolution` when
    `potential` is set; the sign is controlled by `use_negative_gradient`.

    Args:
        x (State): Current state $x \\in \\mathbb{R}^{d_x}$.
        u (Control | None): Current control input $u \\in \\mathbb{R}^{d_u}$ or None.
        t (Time): Current time.

    Returns:
        jax.Array: Scalar potential value $V(x, u, t) \\in \\mathbb{R}$.

    Note:
        This is a protocol interface; implement this callable signature; do not instantiate.
        We recommend simply using a plain Python function that matches this signature, e.g.:

        ```python
        def potential(x, u, t):
            return x[0]**2 + x[1]**2 + x[2]**2
        ```
        or `lambda x, u, t: x[0]**2 + x[1]**2 + x[2]**2`
    """

    def __call__(
        self,
        x: State,
        u: Control | None,
        t: Time,
    ) -> jax.Array:
        raise NotImplementedError()


@dataclasses.dataclass
class ContinuousTimeStateEvolution:
    """
    Continuous-time state evolution via stochastic differential equations (SDEs).

    The state evolves according to

    $$
    dx_t = \\bigl[ \\mu(x_t, u_t, t) + s \\, \\nabla_x V(x_t, u_t, t) \\bigr] \\, dt
         + L(x_t, u_t, t) \\, dW_t
    $$

    where $\\mu$ is the drift, $V$ is an optional potential, and $L$ is the diffusion
    coefficient. The sign $s$ is $-1$ when `use_negative_gradient` is True (e.g., for
    Langevin dynamics) and $+1$ otherwise.

    Attributes:
        drift (Drift | None): Drift vector field $\\mu(x, u, t)$.
            Defaults to zero if None.
            At least one of `drift` or `potential` must be non-None.
        potential (Potential | None): Scalar potential $V(x, u, t)$ whose gradient is added to the drift.
            Defaults to zero if None.
            At least one of `drift` or `potential` must be non-None.
        use_negative_gradient (bool): If True, use $-\\nabla_x V$ (e.g., gradient descent on potential);
            otherwise use $+\\nabla_x V$. Default is False.
        diffusion_coefficient (Drift | None): Diffusion coefficient $L(x, u, t)$ mapping to a matrix;
            multiplies the Brownian increment $dW_t$.
            Defaults to zero if None (i.e., deterministic ODE).
        bm_dim (int | None): Dimension of the Brownian motion $W_t$.
            Inferred automatically from the output shape of `diffusion_coefficient`;
            if passed by the user, it must match diffusion_coefficient(...).shape[1].
    """

    drift: Drift | None = None
    potential: Potential | None = None
    use_negative_gradient: bool = False
    diffusion_coefficient: Drift | None = None
    bm_dim: int | None = dataclasses.field(default=None, repr=False)

    def total_drift(self, x: State, u: Control | None, t: Time) -> dState:
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


class DiscreteTimeStateEvolution:
    """
    Discrete-time state evolution via Markov transition distributions.

    The next state is drawn from a conditional distribution given the current state,
    control, and time indices:

    $$
    x_{t_{k+1}} \\sim p\\left(x_{t_{k+1}} \\mid x_{t_k}, u_{t_k}, t_k, t_{k+1}\\right)
    $$

    Implementations must return a NumPyro-compatible distribution (e.g.,
    `numpyro.distributions.Distribution`) that can be sampled and evaluated.

    Args:
        x (State): Current state $x \\in \\mathbb{R}^{d_x}$.
        u (Control | None): Current control input or None.
        t_now (Time): Current time index $t_k$.
        t_next (Time): Next time index $t_{k+1}$ (for non-uniform sampling or continuous-time embeddings).

    Returns:
        DistributionT: Distribution over the next state $x_{t_{k+1}}$.
            In practice this should be a `numpyro.distributions.Distribution` instance.
    """

    def __call__(
        self,
        x: State,
        u: Control | None,
        t_now: Time,
        t_next: Time,
    ) -> DistributionT:
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

    def log_prob(self, y, x=None, u=None, t=None, *args, **kwargs):
        dist = self(x, u, t)
        return dist.log_prob(y)

    def sample(self, x, u, t, *args, **kwargs):
        dist = self(x, u, t)
        if "seed" in kwargs:  # for CD-Dynamax compatibility
            seed = kwargs.pop("seed")
            kwargs["key"] = seed
        return dist.sample(*args, **kwargs)

    @abstractmethod
    def __call__(self, x, u, t) -> DistributionT: ...

    def masked_log_prob(
        self,
        y: jax.Array,
        obs_mask: jax.Array,
        x: Any,
        u: Any = None,
        t: Any = None,
    ) -> jax.Array:
        """Log p(y_obs | x) scoring only observed dimensions.

        Args:
            y: Observation with NaN replaced by safe values. Shape (obs_dim,).
            obs_mask: Boolean array, True = observed. Shape (obs_dim,).
            x: Latent state.
            u: Control or None.
            t: Time or None.

        Returns:
            Scalar log-probability summed over observed dims only.
        """
        import numpyro.distributions as _dist_mod

        d = self(x, u, t)
        # Unwrap Independent(base, 1) to get per-element log_probs
        base = d
        if isinstance(d, _dist_mod.Independent) and d.reinterpreted_batch_ndims == 1:
            base = d.base_dist
        per_dim_lp = base.log_prob(y)  # (obs_dim,) if base is element-wise
        if jnp.ndim(per_dim_lp) == 0:
            raise NotImplementedError(
                f"{type(self).__name__}.masked_log_prob: distribution "
                f"{type(d).__name__} does not decompose per-dimension. "
                "Override masked_log_prob in the subclass."
            )
        return jnp.sum(jnp.where(obs_mask, per_dim_lp, 0.0))


class WithPartialMissingnessSupport:
    """Mixin for DynamicalModels with particle birth/death semantics.

    Models that mix this class into their dynamics object can provide a static
    latent mask that tells the simulator which state dimensions are "alive"
    (should be sampled from the transition) at each time step.

    The simulator checks ``isinstance(dynamics, WithPartialMissingnessSupport)``
    and calls ``compute_latent_mask`` to obtain the per-timestep mask before the
    scan.  When the mixin is absent, all latent dimensions are always alive.

    Example usage::

        class MyParticleDynamics(DynamicalModel, WithPartialMissingnessSupport):
            def compute_latent_mask(self, obs_values):
                # obs_values: (T, state_dim) with NaN for absent particles
                import numpy as np
                T, N = obs_values.shape
                mask = np.zeros((T, N), dtype=bool)
                for i in range(N):
                    present = ~np.isnan(obs_values[:, i])
                    if present.any():
                        first = np.argmax(present)
                        last = len(present) - 1 - np.argmax(present[::-1])
                        mask[first:last + 1, i] = True
                return mask
    """

    def compute_latent_mask(self, obs_values) -> jax.Array:
        """Return bool array (T, state_dim). True = sample this dim at this time step.

        Args:
            obs_values: Observation array of shape (T, obs_dim), may contain NaN.

        Returns:
            Boolean numpy array of shape (T, state_dim).
        """
        raise NotImplementedError
