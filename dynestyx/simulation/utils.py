"""Shared utilities for simulation backends and handlers."""

import dataclasses
from collections.abc import Callable
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
from jax import Array

from dynestyx.models import DynamicalModel
from dynestyx.types import SimulatedResult, chain_numpyro_site_registrations


def _tile_times(times: Array, n_sim: int) -> Array:
    """Return times tiled to shape (n_sim, T)."""
    return jnp.broadcast_to(jnp.expand_dims(times, axis=0), (n_sim, len(times)))


def _ensure_trailing_dim(arr: Array) -> Array:
    """Ensure simulator outputs follow shape (n_sim, T, dim)."""
    return arr[..., jnp.newaxis] if arr.ndim == 2 else arr


def _merge_segments(
    arr_list: list[Array],
    seg_masks: list[Array],
    n_pred: int,
) -> Array:
    """Merge segment outputs into one array in predict-time order."""
    first = arr_list[0]
    assert first.ndim == 3, (
        f"_merge_segments expects ndim==3 arrays (n_sim, T, D), got ndim={first.ndim} "
        f"with shape {first.shape}. Ensure _ensure_trailing_dim is applied before "
        "calling this function."
    )
    out = jnp.zeros((first.shape[0], n_pred, first.shape[2]), dtype=first.dtype)
    for arr, mask in zip(arr_list, seg_masks, strict=True):
        local_idx = jnp.where(mask, jnp.cumsum(mask) - 1, 0)
        gathered = arr[:, local_idx, :]
        out = jnp.where(mask[None, :, None], gathered, out)
    return out


def _stack_simulated_results(
    results: list[SimulatedResult],
    *,
    plate_shapes: tuple[int, ...],
) -> SimulatedResult:
    """Stack per-member simulation results back onto the plate grid.
    First stacks all array-valued fields, then chains the site registrations.
    """
    stacked_fields = {}
    for field in dataclasses.fields(results[0]):
        values = [getattr(result, field.name) for result in results]
        array_mask = [eqx.is_array(value) for value in values]
        if any(array_mask) and not all(array_mask):
            raise ValueError(
                "Plate simulator members returned inconsistent result fields."
            )
        if not all(array_mask):
            continue

        arrays = cast(list[Array], values)
        stacked_fields[field.name] = jnp.stack(arrays).reshape(
            *plate_shapes, *arrays[0].shape
        )

    return SimulatedResult(
        **stacked_fields,
        _register_numpyro_sites=chain_numpyro_site_registrations(
            *(result._register_numpyro_sites for result in results)
        ),
    )


def _register_simulated_result_sites(
    result: SimulatedResult, *, site_name: str
) -> None:
    """Register a simulation result's populated fields as deterministic sites."""
    for field in dataclasses.fields(result):
        value = getattr(result, field.name)
        if eqx.is_array(value):
            numpyro.deterministic(f"{site_name}_{field.name}", value)


def _sample_initial_states(
    initial_condition,
    *,
    rng_key: Array,
    n_simulations: int,
) -> Array:
    """Draw independent initial states for each simulation member."""
    keys = jr.split(rng_key, n_simulations)
    return jax.vmap(initial_condition.sample)(keys)


def _sample_observation_path(
    dynamics: DynamicalModel,
    *,
    states: Array,
    times: Array,
    rng_key: Array,
    control_path_eval: Callable[[Array], Array | None] | None = None,
) -> Array:
    """Sample one observation path conditional on a realized state path."""
    ctrl = control_path_eval if control_path_eval is not None else (lambda t: None)
    obs_keys = jr.split(rng_key, len(times))

    def _sample_at_time(t_idx):
        x_t = states[t_idx]
        t = times[t_idx]
        obs_dist = dynamics.observation_model(x=x_t, u=ctrl(t), t=t)
        return obs_dist.sample(obs_keys[t_idx])

    return jax.vmap(_sample_at_time)(jnp.arange(len(times)))
