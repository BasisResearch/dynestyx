"""Plate-splitting helpers for latent-path inference."""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from dynestyx.inference.utils.plate_utils import (
    _slice_array_for_plate_member,
    _slice_dist_for_plate_member,
)
from dynestyx.simulation.base import _slice_tree_for_plate_member
from dynestyx.types import LatentStateResult
from dynestyx.utils import _dist_has_plate_batch_dims


@dataclasses.dataclass
class _LatentPlateMemberSpec:
    """One plated latent-path subproblem after slicing plate axes.

    ``LatentPathBuilder`` handles plated requests by splitting them into a list
    of independent single-trajectory subproblems, running each one through the
    scalar latent-path pipeline, and then stacking the results back together.
    This dataclass records one such per-member subproblem.
    """

    name: str
    dynamics: Any
    obs_times: Array | None
    obs_values: Array | None
    obs_values_filled: Array | None
    obs_mask: Array | None
    ctrl_times: Array | None
    ctrl_values: Array | None
    state_path_params: Array | None
    missing_obs_values: Array | None


def _plate_member_specs(
    *,
    name: str,
    dynamics,
    plate_shapes,
    obs_times: Array | None,
    obs_values: Array | None,
    obs_values_filled: Array | None,
    obs_mask: Array | None,
    ctrl_times: Array | None,
    ctrl_values: Array | None,
    state_path_params: Array | None,
    missing_obs_values: Array | None,
) -> list[_LatentPlateMemberSpec]:
    """Slice one plated latent-path request into per-member subproblems.

    This helper mirrors the plate-splitting strategy used elsewhere in the
    codebase: every leading plate index becomes an independent single-path
    latent inference problem with matching slices of dynamics, observations,
    controls, and any user-supplied latent values.
    """
    member_specs = []
    for plate_idx in itertools.product(*[range(s) for s in plate_shapes]):
        member_dynamics = _slice_tree_for_plate_member(
            dynamics, plate_shapes, plate_idx
        )
        if _dist_has_plate_batch_dims(dynamics.initial_condition, plate_shapes):
            member_initial_condition = _slice_dist_for_plate_member(
                dynamics.initial_condition, plate_shapes, plate_idx
            )
            member_dynamics = eqx.tree_at(
                lambda m: m.initial_condition,
                member_dynamics,
                member_initial_condition,
                is_leaf=lambda x: x is None,
            )

        member_specs.append(
            _LatentPlateMemberSpec(
                name=f"{name}_p{'_'.join(str(i) for i in plate_idx)}",
                dynamics=member_dynamics,
                obs_times=_slice_array_for_plate_member(
                    obs_times, plate_shapes, plate_idx
                ),
                obs_values=_slice_array_for_plate_member(
                    obs_values, plate_shapes, plate_idx
                ),
                obs_values_filled=_slice_array_for_plate_member(
                    obs_values_filled, plate_shapes, plate_idx
                ),
                obs_mask=_slice_array_for_plate_member(
                    obs_mask, plate_shapes, plate_idx
                ),
                ctrl_times=_slice_array_for_plate_member(
                    ctrl_times, plate_shapes, plate_idx
                ),
                ctrl_values=_slice_array_for_plate_member(
                    ctrl_values, plate_shapes, plate_idx
                ),
                state_path_params=_slice_array_for_plate_member(
                    state_path_params, plate_shapes, plate_idx
                ),
                missing_obs_values=_slice_array_for_plate_member(
                    missing_obs_values, plate_shapes, plate_idx
                ),
            )
        )
    return member_specs


def _stack_member_attr(
    member_results: list[LatentStateResult], attr: str, plate_shapes
):
    """Stack one latent-result attribute back onto the plate shape.

    Every plated latent-path request is evaluated member-by-member. This helper
    reverses that flattening by restoring the leading plate axes for one result
    attribute at a time. If any member lacks the attribute, the stacked result
    is reported as ``None``.
    """
    values = [getattr(result, attr) for result in member_results]
    if any(value is None for value in values):
        return None
    return jnp.stack([jnp.asarray(value) for value in values]).reshape(
        *plate_shapes, *jnp.asarray(values[0]).shape
    )


__all__ = ["_LatentPlateMemberSpec", "_plate_member_specs", "_stack_member_attr"]
