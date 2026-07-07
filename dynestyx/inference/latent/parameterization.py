"""Latent path parameterizations and state-path reconstruction helpers.

This module defines the map from latent inference variables to full state
trajectories.

Throughout the latent-path code, we distinguish:

- ``z = state_path_params``: the free parameters inferred by an outer
  optimizer/MCMC routine, and
- ``x = state_path = g(z)``: the fully specified state trajectory used to
  evaluate ``log p(x, y | ...)``.

Different models and observation structures induce different choices of
``z``. For example, a discrete model may use one latent state per observation
time, whereas a Dirac observation model with partial missingness may compress
``z`` down to only the unobserved coordinates.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from jax import Array

from dynestyx.models import (
    DeterministicContinuousTimeStateEvolution,
    DiracIdentityObservation,
    DynamicalModel,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
    _canonicalize_observation_distribution,
    _concrete_observation_mask,
    _marginalizable_distribution_mode,
    _probe_observation_distribution,
    _supports_missing_observation_augmentation,
    prepare_missing_observation_metadata,
    prepare_observation_views,
    resolve_missing_observation_strategy,
)
from dynestyx.solvers import solve_ode
from dynestyx.utils import _build_control_path, _raise_now_or_error_if


@dataclasses.dataclass
class DiracLatentMetadata:
    """Concrete indexing metadata for exact-observation latent compression.

    For ``DiracIdentityObservation`` models, observed coordinates are fixed
    exactly by the data, so they do not need corresponding free latent
    parameters. This object records the indexing needed to compress the latent
    parameterization down to only the missing coordinates and later expand that
    compressed vector back into the full state path.
    """

    state_path_param_times: Array
    state_path_param_coordinate_indices: Array
    free_flat_indices: Array
    state_shape: tuple[int, ...]


@dataclasses.dataclass
class AssembledStatePath:
    """Reconstructed latent state path ``x = g(z)`` for one parameterization.

    Writing ``z = state_path_params`` and ``x = state_path = g(z)``, this
    object stores both the free latent variables ``z`` and the full state path
    ``x`` used by the model density ``log p(x, y | ...)``.
    """

    state_path_params: Array
    state_path_param_times: Array
    state_path_param_coordinate_indices: Array | None
    state_path: Array
    state_path_times: Array


def canonicalize_state_path_params(
    dynamics: DynamicalModel,
    state_path_params: Array,
    *,
    n_times: int,
) -> Array:
    """Canonicalize dense ``state_path_params`` so time is the leading axis.

    For discrete or discretized models, ``state_path_params`` are represented as
    a dense array whose leading axis indexes ``state_path_param_times``. For an
    ODE latent path there is only one parameter time, so callers may pass
    either a single state-shaped value or a length-1 leading time axis.
    """
    params = jnp.asarray(state_path_params)
    event_ndim = len(dynamics.initial_condition.event_shape)

    if n_times == 1:
        if params.ndim == event_ndim:
            return jnp.expand_dims(params, axis=0)
        if params.ndim == event_ndim + 1 and params.shape[0] == 1:
            return params
        raise ValueError(
            "state_path_params is incompatible with state_path_param_times. "
            "For a single parameter time, provide either one path parameter or a "
            "length-1 leading time axis."
        )

    if params.ndim < 1 or params.shape[0] != n_times:
        raise ValueError(
            "state_path_params must have a leading time axis matching "
            "state_path_param_times for discrete / discretized models."
        )
    return params


def canonicalize_dirac_state_path_params(
    state_path_params: Array,
    *,
    n_state_path_params: int,
) -> Array:
    """Canonicalize compressed exact-observation path params to a flat vector.

    In the compressed Dirac case, ``state_path_params`` are no longer a dense
    trajectory array. Instead they are a one-dimensional vector containing only
    the free coordinates not fixed by exact observations. This helper enforces
    that convention.
    """
    params = jnp.asarray(state_path_params)

    if n_state_path_params == 0:
        if params.size != 0:
            raise ValueError(
                "This exact-observation trajectory has no free state_path_params. "
                "Provide an empty state_path_params vector."
            )
        return jnp.reshape(params, (0,))

    if params.ndim == 0 and n_state_path_params == 1:
        return jnp.reshape(params, (1,))

    if params.ndim != 1 or params.shape[0] != n_state_path_params:
        raise ValueError(
            "Compressed exact-observation state_path_params must be a flat vector "
            "whose length matches the number of free state coordinates."
        )
    return params


def infer_state_path_param_times(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
) -> Array:
    """Infer the time index attached to ``state_path_params``.

    Current conventions are:

    - discrete / discretized models: one parameter time per observation time,
    - deterministic continuous-time models: a single parameter time at ``t0``
      because the only free latent parameter is the initial condition.
    """
    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if dynamics.t0 is None:
            raise ValueError(
                "Deterministic continuous-time latent-state assembly requires "
                "dynamics.t0 to be resolved before inferring path parameter times."
            )
        obs_times_arr = jnp.asarray(obs_times)
        return jnp.asarray([jnp.asarray(dynamics.t0, dtype=obs_times_arr.dtype)])
    return jnp.asarray(obs_times)


def infer_dirac_state_path_param_metadata(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_mask: Array,
) -> DiracLatentMetadata:
    """Infer compressed latent indexing for exact-observation state paths.

    Mathematically, this chooses a compressed parameterization
    ``z = state_path_params`` containing only the free coordinates of the full
    state path ``x``. The output metadata then defines the scatter operation
    that reconstructs ``x = g(z)`` by filling observed coordinates directly
    from ``obs_values`` and free coordinates from ``z``.
    """
    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        raise ValueError(
            "DiracIdentityObservation missingness compression is not yet "
            "implemented for deterministic continuous-time models."
        )
    if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
        raise ValueError(
            "Latent-state assembly does not yet support native SDE models. "
            "Please discretize the model first."
        )
    if not isinstance(dynamics.observation_model, DiracIdentityObservation):
        raise ValueError(
            "Dirac latent metadata is only defined for DiracIdentityObservation."
        )

    try:
        obs_mask_np = np.asarray(obs_mask, dtype=bool)
        obs_times_np = np.asarray(obs_times)
    except Exception as exc:  # pragma: no cover - defensive for traced callers
        raise ValueError(
            "Dirac latent compression currently requires a concrete observation "
            "missingness pattern. Prepare the latent layout eagerly with "
            "prepare_latent_path_layout(...) or prepare_dirac_state_path_metadata(...)."
        ) from exc

    free_mask_np = ~obs_mask_np
    flat_free_indices_np = np.flatnonzero(free_mask_np.reshape(-1))

    if obs_mask_np.ndim == 1:
        state_path_param_times_np = obs_times_np[free_mask_np]
        coord_indices_np = np.zeros((flat_free_indices_np.shape[0],), dtype=np.int32)
    elif obs_mask_np.ndim == 2:
        time_grid_np = np.broadcast_to(obs_times_np[:, None], obs_mask_np.shape)
        coord_grid_np = np.broadcast_to(
            np.arange(obs_mask_np.shape[-1], dtype=np.int32)[None, :],
            obs_mask_np.shape,
        )
        state_path_param_times_np = time_grid_np[free_mask_np]
        coord_indices_np = coord_grid_np[free_mask_np]
    else:
        raise ValueError(
            "Dirac latent compression expects obs_mask with shape (time,) or "
            "(time, observation_dim)."
        )

    obs_times_arr = jnp.asarray(obs_times)
    return DiracLatentMetadata(
        state_path_param_times=jnp.asarray(
            state_path_param_times_np,
            dtype=obs_times_arr.dtype,
        ),
        state_path_param_coordinate_indices=jnp.asarray(
            coord_indices_np, dtype=jnp.int32
        ),
        free_flat_indices=jnp.asarray(flat_free_indices_np, dtype=jnp.int32),
        state_shape=tuple(obs_mask_np.shape),
    )


def prepare_dirac_state_path_metadata(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_values: Array | None = None,
    obs_mask: Array | None = None,
) -> DiracLatentMetadata:
    """Precompute exact-observation compression metadata outside traced code.

    Dirac compression depends on the concrete missingness pattern. When callers
    are about to enter traced NumPyro/JAX code, they can use this helper
    eagerly so that the latent layout is fixed ahead of time rather than being
    inferred from traced arrays.
    """
    if (obs_values is None) == (obs_mask is None):
        raise ValueError(
            "Provide exactly one of obs_values or obs_mask when preparing Dirac "
            "state-path metadata."
        )

    if obs_mask is None:
        assert obs_values is not None
        concrete_obs_mask = _concrete_observation_mask(obs_values)
        if concrete_obs_mask is not None:
            return infer_dirac_state_path_param_metadata(
                dynamics,
                obs_times=obs_times,
                obs_mask=jnp.asarray(concrete_obs_mask),
            )
        _, obs_mask, _ = prepare_observation_views(dynamics, obs_values)
        if obs_mask is None:
            raise ValueError(
                "Could not prepare an observation mask for Dirac state-path metadata."
            )

    return infer_dirac_state_path_param_metadata(
        dynamics,
        obs_times=obs_times,
        obs_mask=obs_mask,
    )


def fully_observed_dirac_state_path_param_metadata(
    *,
    obs_times: Array,
    state_shape: tuple[int, ...],
) -> DiracLatentMetadata:
    """Return empty compression metadata for fully observed exact observations.

    In this case the exact observations determine the entire state path, so the
    compressed latent vector has length zero.
    """
    obs_times_arr = jnp.asarray(obs_times)
    return DiracLatentMetadata(
        state_path_param_times=jnp.asarray([], dtype=obs_times_arr.dtype),
        state_path_param_coordinate_indices=jnp.asarray([], dtype=jnp.int32),
        free_flat_indices=jnp.asarray([], dtype=jnp.int32),
        state_shape=state_shape,
    )


def default_ode_diffeqsolve_settings() -> dict[str, Any]:
    """Return default solver settings for ODE latent-path reconstruction.

    These settings are only used when reconstructing
    ``x = state_path = g(z)`` for deterministic continuous-time models, where
    ``z`` consists only of the initial condition and the rest of the path is
    generated by solving the ODE forward.
    """
    return {
        "solver": dfx.Tsit5(),
        "stepsize_controller": dfx.ConstantStepSize(),
        "adjoint": dfx.RecursiveCheckpointAdjoint(),
        "dt0": jnp.asarray(1e-3),
        "max_steps": 100_000,
    }


def assemble_dirac_state_path(
    *,
    state_path_params: Array,
    latent_metadata: DiracLatentMetadata,
    obs_times: Array,
    obs_values_filled: Array,
) -> AssembledStatePath:
    """Reconstruct the full state path from compressed Dirac latents.

    Here ``z = state_path_params`` is the vector of free unobserved
    coordinates. The reconstruction ``x = g(z)`` is performed by taking the
    observation-shaped dense array ``obs_values_filled`` and scattering those
    free coordinates back into the locations identified by
    ``latent_metadata.free_flat_indices``.
    """
    canonical_params = canonicalize_dirac_state_path_params(
        state_path_params,
        n_state_path_params=latent_metadata.free_flat_indices.shape[0],
    )
    flat_state_path = jnp.reshape(jnp.asarray(obs_values_filled), (-1,))
    state_path = (
        flat_state_path.at[latent_metadata.free_flat_indices]
        .set(canonical_params)
        .reshape(latent_metadata.state_shape)
    )
    obs_times_arr = jnp.asarray(obs_times)
    return AssembledStatePath(
        state_path_params=canonical_params,
        state_path_param_times=latent_metadata.state_path_param_times,
        state_path_param_coordinate_indices=(
            latent_metadata.state_path_param_coordinate_indices
        ),
        state_path=state_path,
        state_path_times=obs_times_arr,
    )


def assemble_state_path(
    dynamics: DynamicalModel,
    *,
    state_path_params: Array,
    state_path_param_times: Array,
    obs_times: Array | None = None,
    ctrl_times: Array | None = None,
    ctrl_values: Array | None = None,
    ode_diffeqsolve_settings: dict[str, Any] | None = None,
) -> AssembledStatePath:
    """Assemble a full state path ``x = g(z)`` from dense path parameters.

    This is the non-compressed reconstruction path:

    - for discrete / discretized models, the dense latent block already is the
      full state path, so ``g`` is essentially the identity;
    - for deterministic continuous-time models, ``z`` contains only the
      initial condition and ``g`` solves the ODE to obtain the full path at the
      requested times.
    """
    state_path_param_times = jnp.asarray(state_path_param_times)
    _raise_now_or_error_if(
        state_path_param_times,
        state_path_param_times.shape[0] < 1,
        "state_path_param_times must contain at least one time point.",
    )

    canonical_params = canonicalize_state_path_params(
        dynamics,
        state_path_params,
        n_times=state_path_param_times.shape[0],
    )

    if isinstance(dynamics.state_evolution, StochasticContinuousTimeStateEvolution):
        raise ValueError(
            "Latent-state assembly does not yet support native SDE models. "
            "Please discretize the model first."
        )

    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        if state_path_param_times.shape[0] != 1:
            raise ValueError(
                "Deterministic continuous-time models expect exactly one latent "
                "path parameter: the initial condition."
            )

        if obs_times is None:
            state_path_times = state_path_param_times
            state_path = canonical_params
        else:
            state_path_times = jnp.concatenate(
                [state_path_param_times, jnp.asarray(obs_times)],
                axis=0,
            )
            if ctrl_times is not None and ctrl_values is not None:
                control_path = _build_control_path(
                    ctrl_times, ctrl_values, state_path_times
                )
                control_path_eval: Callable[[Array], Array | None] = lambda t: (
                    control_path.evaluate(t, left=False)
                )
            else:
                control_path_eval = lambda t: None

            state_path = solve_ode(
                dynamics,
                t0=state_path_param_times[0],
                saveat_times=state_path_times,
                x0=canonical_params[0],
                control_path_eval=control_path_eval,
                diffeqsolve_settings=(
                    ode_diffeqsolve_settings
                    if ode_diffeqsolve_settings is not None
                    else default_ode_diffeqsolve_settings()
                ),
            )

        return AssembledStatePath(
            state_path_params=canonical_params,
            state_path_param_times=state_path_param_times,
            state_path_param_coordinate_indices=None,
            state_path=state_path,
            state_path_times=state_path_times,
        )

    return AssembledStatePath(
        state_path_params=canonical_params,
        state_path_param_times=state_path_param_times,
        state_path_param_coordinate_indices=None,
        state_path=canonical_params,
        state_path_times=state_path_param_times,
    )


@dataclasses.dataclass
class StatePathParameterization:
    """Concrete definition of latent variables ``z`` and reconstruction ``x=g(z)``.

    For one concrete model and observation configuration, this object defines:

    - which free latent variables ``z = state_path_params`` are used,
    - which times / coordinates index those latent variables,
    - whether observation-side augmentation is active, and
    - how to reconstruct the full latent path ``x = state_path = g(z)``.

    This is the central "layout" object for latent inference. Once it is fixed,
    the rest of the latent-path pipeline can treat reconstruction and scoring as
    pure numerical computations rather than API decisions.
    """

    state_path_param_times: Array
    dirac_metadata: DiracLatentMetadata | None = None
    missing_obs_metadata: MissingObservationMetadata | None = None
    uses_missing_obs_augmentation: bool = False
    exact_observation_mask: Array | None = None
    uses_dense_missing_obs_augmentation: bool = False
    dense_missing_obs_shape: tuple[int, ...] | None = None

    @property
    def state_path_param_coordinate_indices(self) -> Array | None:
        """Return coordinate indices for compressed parameterizations when present.

        This is only meaningful for compressed Dirac layouts, where
        ``state_path_params`` is indexed not just by time but by
        ``(time, coordinate)`` pairs.
        """
        if self.dirac_metadata is None:
            return None
        return self.dirac_metadata.state_path_param_coordinate_indices

    def canonicalize_state_path_params(
        self,
        dynamics: DynamicalModel,
        state_path_params: Array,
    ) -> Array:
        """Canonicalize ``z = state_path_params`` for this parameterization.

        The concrete shape rules depend on the active layout:

        - compressed Dirac layouts expect a flat vector of free coordinates,
        - dense layouts expect a leading time axis matching
          ``state_path_param_times``.
        """
        if self.dirac_metadata is not None:
            return canonicalize_dirac_state_path_params(
                state_path_params,
                n_state_path_params=self.dirac_metadata.free_flat_indices.shape[0],
            )
        return canonicalize_state_path_params(
            dynamics,
            state_path_params,
            n_times=self.state_path_param_times.shape[0],
        )

    def example_state_path_params(self, dynamics: DynamicalModel) -> Array:
        """Return a shape-only example latent block for NumPyro site setup.

        This does not represent a meaningful latent value. Its only purpose is
        to define the correct event shape for dummy NumPyro sample sites when
        the actual latent values are not yet available.
        """
        if self.dirac_metadata is not None:
            return jnp.zeros((self.dirac_metadata.free_flat_indices.shape[0],))
        return canonicalize_state_path_params(
            dynamics,
            jnp.zeros(
                (
                    self.state_path_param_times.shape[0],
                    *dynamics.initial_condition.event_shape,
                )
            ),
            n_times=self.state_path_param_times.shape[0],
        )

    def assemble_state_path(
        self,
        dynamics: DynamicalModel,
        *,
        state_path_params: Array,
        obs_times: Array,
        obs_values_filled: Array | None,
        ctrl_times: Array | None = None,
        ctrl_values: Array | None = None,
        ode_diffeqsolve_settings: dict[str, Any] | None = None,
    ) -> AssembledStatePath:
        """Reconstruct the full latent state path ``x = g(z)``.

        There are three main reconstruction modes:

        - compressed Dirac layout: fill observed coordinates from the data and
          free coordinates from ``z``,
        - exact-observation mask fallback: use a dense state-shaped ``z`` but
          overwrite exactly observed entries with data,
        - standard dense layout: delegate to :func:`assemble_state_path`.
        """
        if self.dirac_metadata is not None:
            if obs_values_filled is None:
                raise ValueError(
                    "Dirac latent-path assembly requires pre-filled observation values."
                )
            return assemble_dirac_state_path(
                state_path_params=state_path_params,
                latent_metadata=self.dirac_metadata,
                obs_times=obs_times,
                obs_values_filled=obs_values_filled,
            )

        if self.exact_observation_mask is not None:
            if obs_values_filled is None:
                raise ValueError(
                    "Exact-observation latent-state assembly requires pre-filled "
                    "observation values."
                )
            if jnp.asarray(state_path_params).size == 0:
                return AssembledStatePath(
                    state_path_params=jnp.asarray(state_path_params),
                    state_path_param_times=self.state_path_param_times,
                    state_path_param_coordinate_indices=None,
                    state_path=jnp.asarray(obs_values_filled),
                    state_path_times=jnp.asarray(obs_times),
                )
            canonical_params = canonicalize_state_path_params(
                dynamics,
                state_path_params,
                n_times=self.state_path_param_times.shape[0],
            )
            state_path = jnp.where(
                self.exact_observation_mask,
                jnp.asarray(obs_values_filled),
                canonical_params,
            )
            return AssembledStatePath(
                state_path_params=canonical_params,
                state_path_param_times=self.state_path_param_times,
                state_path_param_coordinate_indices=None,
                state_path=state_path,
                state_path_times=jnp.asarray(obs_times),
            )

        return assemble_state_path(
            dynamics,
            state_path_params=state_path_params,
            state_path_param_times=self.state_path_param_times,
            obs_times=obs_times,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            ode_diffeqsolve_settings=ode_diffeqsolve_settings,
        )


def prepare_state_path_parameterization(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_values: Array,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    obs_values_filled: Array | None = None,
    obs_mask: Array | None = None,
    obs_has_missing: bool | None = None,
) -> StatePathParameterization:
    """Prepare the concrete latent parameterization for one observation setup.

    This is the main layout-selection routine. Given a model, observation
    times, observation values, and a missingness strategy, it decides which
    latent representation should be used for inference.

    In broad strokes:

    - standard models use dense ``state_path_params`` indexed by
      ``state_path_param_times``,
    - ``DiracIdentityObservation`` models may compress away observed
      coordinates,
    - missing observations may additionally activate explicit
      ``missing_obs_values`` augmentation.

    The returned :class:`StatePathParameterization` is then reused throughout
    the builder, scorer, and NumPyro-registration code paths.
    """
    obs_times_arr = jnp.asarray(obs_times)
    obs_values_arr = jnp.asarray(obs_values)

    if obs_values_filled is None or obs_mask is None or obs_has_missing is None:
        obs_values_filled, obs_mask, obs_has_missing = prepare_observation_views(
            dynamics, obs_values_arr
        )

    is_dirac_observation = isinstance(
        dynamics.observation_model, DiracIdentityObservation
    )
    is_ode = isinstance(
        dynamics.state_evolution, DeterministicContinuousTimeStateEvolution
    )
    is_sde = isinstance(
        dynamics.state_evolution, StochasticContinuousTimeStateEvolution
    )
    use_dirac_compression = (
        is_dirac_observation
        and not is_ode
        and not is_sde
        and obs_values_filled is not None
        and (obs_has_missing is False or obs_mask is not None)
    )
    if (
        use_dirac_compression
        and missing_observation_strategy == "augment"
        and obs_has_missing is not False
    ):
        raise ValueError(
            "DiracIdentityObservation missingness should be handled via "
            "state-path compression, not explicit missing-observation augmentation."
        )

    dirac_metadata = None
    exact_observation_mask = None
    if use_dirac_compression:
        if obs_has_missing is False:
            dirac_metadata = fully_observed_dirac_state_path_param_metadata(
                obs_times=obs_times_arr,
                state_shape=tuple(jnp.asarray(obs_values_filled).shape),
            )
        else:
            try:
                dirac_metadata = prepare_dirac_state_path_metadata(
                    dynamics,
                    obs_times=obs_times_arr,
                    obs_values=obs_values_arr,
                )
            except ValueError as exc:
                if obs_mask is None:
                    raise
                if "concrete observation missingness pattern" not in str(exc):
                    raise
                exact_observation_mask = obs_mask

    state_path_param_times = (
        dirac_metadata.state_path_param_times
        if dirac_metadata is not None
        else infer_state_path_param_times(dynamics, obs_times=obs_times_arr)
    )
    if exact_observation_mask is not None:
        state_path_param_times = obs_times_arr

    missing_obs_metadata = None
    uses_missing_obs_augmentation = False
    uses_dense_missing_obs_augmentation = False
    dense_missing_obs_shape = None
    if (
        dirac_metadata is None
        and exact_observation_mask is None
        and missing_observation_strategy in ("augment", "auto")
        and obs_mask is not None
    ):
        observation_dim = 1 if obs_values_arr.ndim == 1 else obs_values_arr.shape[-1]
        probed_obs_dist = _canonicalize_observation_distribution(
            _probe_observation_distribution(dynamics),
            observation_dim=observation_dim,
        )
        marginal_mode = _marginalizable_distribution_mode(probed_obs_dist)
        augmentation_supported = _supports_missing_observation_augmentation(
            probed_obs_dist
        )

        if missing_observation_strategy == "auto" and marginal_mode is not None:
            uses_missing_obs_augmentation = False
        else:
            try:
                effective_missing_obs_metadata = prepare_missing_observation_metadata(
                    dynamics,
                    obs_times=obs_times_arr,
                    obs_values=obs_values_arr,
                )
            except ValueError as exc:
                if "concrete missingness pattern" not in str(exc):
                    raise
                if missing_observation_strategy == "augment":
                    if not augmentation_supported:
                        raise NotImplementedError(
                            "Explicit missing-observation augmentation currently "
                            "requires a continuous observation family."
                        ) from exc
                    uses_missing_obs_augmentation = True
                    uses_dense_missing_obs_augmentation = True
                    dense_missing_obs_shape = tuple(obs_values_arr.shape)
                elif missing_observation_strategy == "auto":
                    uses_missing_obs_augmentation = False
                else:
                    raise
            else:
                if effective_missing_obs_metadata.observation_shape != tuple(
                    obs_values_arr.shape
                ):
                    raise ValueError(
                        "Prepared missing observation metadata does not match the "
                        "observed data shape for this latent-path parameterization."
                    )

                uses_missing_obs_augmentation, _ = resolve_missing_observation_strategy(
                    dynamics,
                    observation_dim=observation_dim,
                    has_missing=effective_missing_obs_metadata.has_missing,
                    has_partial_missing=(
                        effective_missing_obs_metadata.has_partial_missing
                    ),
                    requested_strategy=missing_observation_strategy,
                )
                if uses_missing_obs_augmentation:
                    missing_obs_metadata = effective_missing_obs_metadata

    return StatePathParameterization(
        state_path_param_times=state_path_param_times,
        dirac_metadata=dirac_metadata,
        missing_obs_metadata=missing_obs_metadata,
        uses_missing_obs_augmentation=uses_missing_obs_augmentation,
        exact_observation_mask=exact_observation_mask,
        uses_dense_missing_obs_augmentation=uses_dense_missing_obs_augmentation,
        dense_missing_obs_shape=dense_missing_obs_shape,
    )


def prepare_latent_path_layout(
    dynamics: DynamicalModel,
    *,
    obs_times: Array,
    obs_values: Array,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    obs_values_filled: Array | None = None,
    obs_mask: Array | None = None,
    obs_has_missing: bool | None = None,
) -> StatePathParameterization:
    """Backward-compatible wrapper for eager latent parameterization prep."""
    return prepare_state_path_parameterization(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        missing_observation_strategy=missing_observation_strategy,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        obs_has_missing=obs_has_missing,
    )


LatentPathLayout = StatePathParameterization
AssembledStateTrajectory = AssembledStatePath


__all__ = [
    "AssembledStatePath",
    "AssembledStateTrajectory",
    "DiracLatentMetadata",
    "LatentPathLayout",
    "StatePathParameterization",
    "assemble_dirac_state_path",
    "assemble_state_path",
    "canonicalize_dirac_state_path_params",
    "canonicalize_state_path_params",
    "default_ode_diffeqsolve_settings",
    "fully_observed_dirac_state_path_param_metadata",
    "infer_dirac_state_path_param_metadata",
    "infer_state_path_param_times",
    "prepare_dirac_state_path_metadata",
    "prepare_latent_path_layout",
    "prepare_state_path_parameterization",
]
