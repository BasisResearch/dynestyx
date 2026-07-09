"""Request preparation helpers for latent-path inference.

This module turns a user-facing ``LatentPathBuilder`` request into a concrete
internal problem description. By the time these helpers return, the handler
knows the latent layout, any directly supplied latent values, and the
shape-only example values needed for later NumPyro site registration.
"""

from __future__ import annotations

import dataclasses

from jax.core import Tracer
from jaxtyping import Array

from dynestyx.inference.state_paths.layout import (
    LatentPathLayout,
    prepare_latent_path_layout,
)
from dynestyx.observation_missingness import MissingObservationStrategy

_LATENT_PATH_LAYOUT_CACHE: dict[str, LatentPathLayout] = {}


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

    layout: LatentPathLayout
    obs_values_filled: Array | None
    obs_mask: Array | None
    canonical_state_path_params: Array | None
    canonical_missing_obs_values: Array | None
    example_state_path_params: Array
    example_missing_obs_values: Array | None


def _prepare_missing_obs_values(
    *,
    layout: LatentPathLayout,
    missing_obs_values: Array | None,
) -> tuple[Array | None, Array | None]:
    """Canonicalize or synthesize the missing-observation latent block.

    Returns a pair ``(canonical, example)``:

    - ``canonical`` is the directly supplied latent value, when present.
    - ``example`` is the shape-only placeholder used to define NumPyro sample
      sites under ``dsx.sample(...)``.

    For explicit augmentation, the latent block may either be the flat vector
    of missing coordinates or a dense observation-shaped block when the layout
    chose dense augmentation. Exact-observation state assembly does not create
    a separate ``missing_obs_values`` block here; its missing coordinates are
    handled through ``state_path_params`` instead.
    """
    example_missing_obs_values = layout.example_missing_obs_values()
    if missing_obs_values is None:
        return None, example_missing_obs_values

    if example_missing_obs_values is None:
        raise ValueError(
            "missing_obs_values was provided, but this latent-path layout does "
            "not use a separate missing_obs_values block."
        )

    canonical_missing_obs_values = layout.canonicalize_missing_obs_values(
        missing_obs_values
    )
    return canonical_missing_obs_values, canonical_missing_obs_values


def _prepare_latent_path_request(
    *,
    name: str,
    dynamics,
    obs_times: Array | None,
    obs_values: Array | None,
    obs_values_filled: Array | None,
    obs_mask: Array | None,
    obs_has_missing: bool | None,
    state_path_params: Array | None,
    missing_obs_values: Array | None,
    missing_observation_strategy: MissingObservationStrategy,
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
    if obs_times is None or obs_values is None:
        raise ValueError(
            "LatentPathBuilder requires obs_times and obs_values. "
            "It is an observation-consuming handler."
        )

    if name in _LATENT_PATH_LAYOUT_CACHE and (
        isinstance(obs_values, Tracer)
        or isinstance(obs_mask, Tracer)
        or isinstance(obs_times, Tracer)
    ):
        layout = _LATENT_PATH_LAYOUT_CACHE[name]
    else:
        layout = prepare_latent_path_layout(
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            missing_observation_strategy=missing_observation_strategy,
            obs_values_filled=obs_values_filled,
            obs_mask=obs_mask,
            obs_has_missing=obs_has_missing,
        )
        _LATENT_PATH_LAYOUT_CACHE[name] = layout

    canonical_missing_obs_values, example_missing_obs_values = (
        _prepare_missing_obs_values(
            layout=layout,
            missing_obs_values=missing_obs_values,
        )
    )

    canonical_state_path_params = None
    if state_path_params is not None:
        canonical_state_path_params = layout.canonicalize_state_path_params(
            dynamics, state_path_params
        )

    return _PreparedLatentPathRequest(
        layout=layout,
        obs_values_filled=obs_values_filled,
        obs_mask=obs_mask,
        canonical_state_path_params=canonical_state_path_params,
        canonical_missing_obs_values=canonical_missing_obs_values,
        example_state_path_params=(
            canonical_state_path_params
            if canonical_state_path_params is not None
            else layout.example_state_path_params(dynamics)
        ),
        example_missing_obs_values=example_missing_obs_values,
    )


__all__ = ["_PreparedLatentPathRequest", "_prepare_latent_path_request"]
