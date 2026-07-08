"""Request preparation helpers for latent-path inference.

This module turns a user-facing ``LatentPathBuilder`` request into a concrete
internal problem description. By the time these helpers return, the handler
knows the latent layout, any directly supplied latent values, and the
shape-only example values needed for later NumPyro site registration.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
from jaxtyping import Array

from dynestyx.inference.latent.parameterization import (
    StatePathParameterization,
    prepare_state_path_parameterization,
)
from dynestyx.observation_missingness import (
    MissingObservationStrategy,
    canonicalize_missing_obs_values,
)


@dataclasses.dataclass
class _PreparedLatentPathRequest:
    """Concrete latent-path inputs prepared ahead of evaluation/registration.

    This is the boundary object between request parsing and actual inference
    work. By the time an instance exists, the handler has already decided:

    - which latent layout is active,
    - how user-provided latent values should be canonicalized, and
    - what shape-only example values NumPyro will need if sites are registered.

    The later evaluation and registration steps therefore do not need to repeat
    any layout-resolution logic.
    """

    parameterization: StatePathParameterization
    obs_values_filled: Array | None
    obs_mask: Array | None
    canonical_state_path_params: Array | None
    canonical_missing_obs_values: Array | None
    example_state_path_params: Array
    example_missing_obs_values: Array | None


def _validate_latent_path_request(
    *,
    obs_times: Array | None,
    obs_values: Array | None,
) -> None:
    """Validate the observation inputs required by ``LatentPathBuilder``.

    Unlike simulators, the latent-path handler is always an
    observation-consuming inference object. It therefore requires concrete
    ``obs_times`` and ``obs_values`` so that the latent layout and joint score
    ``log p(x, y | ...)`` are both well-defined.
    """
    if obs_times is None or obs_values is None:
        raise ValueError(
            "LatentPathBuilder requires obs_times and obs_values. "
            "It is an observation-consuming handler."
        )


def _resolve_state_path_parameterization(
    *,
    dynamics,
    obs_times: Array,
    obs_values: Array,
    obs_values_filled: Array | None,
    obs_mask: Array | None,
    obs_has_missing: bool | None,
    latent_observation_mode: MissingObservationStrategy,
    latent_path_layout: StatePathParameterization | None,
) -> StatePathParameterization:
    """Resolve the concrete latent parameterization for one trajectory.

    If the caller supplied a precomputed layout, this helper simply reuses it.
    Otherwise it derives a fresh :class:`StatePathParameterization` from the
    model, times, observations, and missingness strategy.
    """
    if latent_path_layout is not None:
        return latent_path_layout

    return prepare_state_path_parameterization(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        missing_observation_strategy=latent_observation_mode,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        obs_has_missing=obs_has_missing,
    )


def _prepare_missing_obs_values(
    *,
    parameterization: StatePathParameterization,
    missing_obs_values: Array | None,
) -> tuple[Array | None, Array | None]:
    """Canonicalize or synthesize the missing-observation latent block.

    Returns a pair ``(canonical, example)``:

    - ``canonical`` is the directly supplied latent value, when present.
    - ``example`` is the shape-only placeholder used to define NumPyro sample
      sites under ``dsx.sample(...)``.

    For explicit augmentation, the latent block may either be the compressed
    vector of missing coordinates or a dense observation-shaped block when the
    layout chose dense augmentation.
    """
    if parameterization.uses_dense_missing_obs_augmentation:
        assert parameterization.dense_missing_obs_shape is not None
        if missing_obs_values is None:
            return None, jnp.zeros(parameterization.dense_missing_obs_shape)
        dense_missing_obs_values = jnp.asarray(missing_obs_values)
        if (
            tuple(dense_missing_obs_values.shape)
            != parameterization.dense_missing_obs_shape
        ):
            raise ValueError(
                "Dense missing_obs_values must match the observation array shape "
                "for this latent-path parameterization."
            )
        return dense_missing_obs_values, dense_missing_obs_values

    metadata = parameterization.missing_obs_metadata
    if not parameterization.uses_missing_obs_augmentation or metadata is None:
        return None, None

    n_missing_obs = metadata.free_flat_indices.shape[0]
    if missing_obs_values is None:
        return None, jnp.zeros((n_missing_obs,))

    canonical_missing_obs_values = canonicalize_missing_obs_values(
        missing_obs_values,
        n_missing_obs=n_missing_obs,
    )
    return canonical_missing_obs_values, canonical_missing_obs_values


def _prepare_latent_path_request(
    *,
    dynamics,
    obs_times: Array | None,
    obs_values: Array | None,
    obs_values_filled: Array | None,
    obs_mask: Array | None,
    obs_has_missing: bool | None,
    latent_path_layout: StatePathParameterization | None,
    state_path_params: Array | None,
    missing_obs_values: Array | None,
    latent_observation_mode: MissingObservationStrategy,
) -> _PreparedLatentPathRequest:
    """Prepare canonical latent inputs for later evaluation or registration.

    Conceptually this helper freezes the inference request into a concrete
    latent problem:

    - resolve the layout ``z -> x = g(z)``,
    - canonicalize any user-provided ``state_path_params`` or
      ``missing_obs_values``, and
    - construct example latent values for NumPyro site creation when needed.

    The returned object can then be reused by both the eager pure-JAX
    evaluation and the deferred NumPyro registration path.
    """
    _validate_latent_path_request(obs_times=obs_times, obs_values=obs_values)
    assert obs_times is not None
    assert obs_values is not None

    parameterization = _resolve_state_path_parameterization(
        dynamics=dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        obs_has_missing=obs_has_missing,
        latent_observation_mode=latent_observation_mode,
        latent_path_layout=latent_path_layout,
    )
    canonical_missing_obs_values, example_missing_obs_values = (
        _prepare_missing_obs_values(
            parameterization=parameterization,
            missing_obs_values=missing_obs_values,
        )
    )

    canonical_state_path_params = None
    if state_path_params is not None:
        canonical_state_path_params = parameterization.canonicalize_state_path_params(
            dynamics, state_path_params
        )

    return _PreparedLatentPathRequest(
        parameterization=parameterization,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        canonical_state_path_params=canonical_state_path_params,
        canonical_missing_obs_values=canonical_missing_obs_values,
        example_state_path_params=(
            canonical_state_path_params
            if canonical_state_path_params is not None
            else parameterization.example_state_path_params(dynamics)
        ),
        example_missing_obs_values=example_missing_obs_values,
    )


__all__ = ["_PreparedLatentPathRequest", "_prepare_latent_path_request"]
