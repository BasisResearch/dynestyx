"""Pure-JAX state-path scoring helpers."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, Bool, Int, Real, Shaped

from dynestyx.models import DeterministicContinuousTimeStateEvolution, DynamicalModel
from dynestyx.observation_missingness import (
    MissingObservationMetadata,
    MissingObservationStrategy,
    prepare_observation_log_prob,
)
from dynestyx.utils import _get_val_or_None, _raise_now_or_error_if


def _gather_by_exact_time(
    values: Shaped[Array, "source_time value_dim"] | Shaped[Array, " source_time"],
    source_times: Real[Array, " source_time"],
    query_times: Real[Array, " query_time"],
    *,
    value_name: str,
) -> Shaped[Array, "query_time value_dim"] | Shaped[Array, " query_time"]:
    """Select values whose source times exactly match query times.

    `source_times` must be sorted in ascending order, and its leading axis must
    align with the leading axis of `values`. The function uses exact equality;
    it does not interpolate or choose a nearby time. These shape and sorting
    requirements are not checked directly.

    Args:
        values: Values indexed by `source_times`.
        source_times: Sorted times associated with the leading axis of `values`.
        query_times: Times to select.
        value_name: Name used in the error message.

    Returns:
        Array: Selected values in `query_times` order. An empty query returns an
            empty slice of `values`.

    Raises:
        eqx.EquinoxRuntimeError: If any query time is absent from `source_times`.
    """
    source = jnp.asarray(source_times)
    query = jnp.asarray(query_times)
    if query.size == 0:
        return values[:0]

    idx = jnp.searchsorted(source, query, side="left")
    max_idx = source.shape[0] - 1
    safe_idx = jnp.clip(idx, 0, max_idx)
    matched = (idx < source.shape[0]) & (source[safe_idx] == query)
    safe_idx = eqx.error_if(
        safe_idx,
        jnp.any(~matched),
        f"{value_name} must be defined exactly at every requested query time.",
    )
    return values[safe_idx]


def _control_values_at_times(
    ctrl_times: Real[Array, " ctrl_time"] | None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None,
    query_times: Real[Array, " query_time"] | None,
) -> Real[Array, "query_time control_dim"] | Real[Array, " query_time"] | None:
    """Return control values at exact query times.

    Args:
        ctrl_times: Times associated with `ctrl_values`.
        ctrl_values: Control values.
        query_times: Times at which controls are required.

    Returns:
        Array | None: Control values in `query_times` order. Returns `None` if
            any argument is `None`.

    Raises:
        eqx.EquinoxRuntimeError: If a query time is absent from `ctrl_times`.
    """
    if ctrl_times is None or ctrl_values is None or query_times is None:
        return None
    return _gather_by_exact_time(
        jnp.asarray(ctrl_values),
        jnp.asarray(ctrl_times),
        jnp.asarray(query_times),
        value_name="ctrl_values",
    )


def compute_state_path_log_prob(
    dynamics: DynamicalModel,
    *,
    state_path: Real[Array, "state_path_time state_dim"]
    | Real[Array, " state_path_time"],
    state_path_times: Real[Array, " state_path_time"],
    obs_times: Real[Array, " obs_time"] | None = None,
    obs_values: Real[Array, "obs_time observation_dim"]
    | Real[Array, " obs_time"]
    | None = None,
    obs_values_filled: Real[Array, "obs_time observation_dim"]
    | Real[Array, " obs_time"]
    | None = None,
    obs_mask: Bool[Array, "obs_time observation_dim"]
    | Bool[Array, " obs_time"]
    | None = None,
    missing_observation_strategy: MissingObservationStrategy = "auto",
    missing_obs_values: Real[Array, " n_missing_obs"]
    | Real[Array, " obs_time"]
    | Real[Array, "obs_time observation_dim"]
    | Real[Array, ""]
    | None = None,
    missing_obs_metadata: MissingObservationMetadata | None = None,
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"]
    | Real[Array, " ctrl_time"]
    | None = None,
    chunk_size: int | None = 0,
    observations_are_exact_constraints: bool = False,
) -> Real[Array, "*log_prob_batch"]:
    """Evaluate the joint log density of a reconstructed state path.

    If either `obs_times` or `obs_values` is `None`, observation terms
    are omitted.

    Args:
        dynamics: Dynamical model that defines the initial, transition, and
            observation distributions.
        state_path: Complete state values. Its leading axis must align with
            `state_path_times`.
        state_path_times: Times associated with `state_path`. This array must
            contain at least one entry.
        obs_times: Times associated with `obs_values`. Every observation time
            must occur exactly in `state_path_times`.
        obs_values: Observation values, including any missing entries.
        obs_values_filled: Observation values with missing entries replaced by
            shape-preserving filler values.
        obs_mask: Boolean array that marks observed entries.
        missing_observation_strategy: Method used to handle missing entries in
            `obs_values`.
        missing_obs_values: Values used to complete missing observations when
            augmentation is active. Supply either a flat vector ordered by
            `missing_obs_metadata`, a scalar for one missing entry, or a dense
            array shaped like `obs_values`; observed entries in a dense array
            are ignored.
        missing_obs_metadata: Positions, times, and component indices for
            `missing_obs_values`.
        ctrl_times: Times associated with `ctrl_values`. Required control times
            must occur exactly in this array.
        ctrl_values: Control values, or `None` for an uncontrolled model.
        chunk_size: Batch size passed to `jax.lax.map` while scoring transition
            and observation terms. The default, `0`, evaluates all terms with
            one `jax.vmap`. `None` maps one term at a time. A positive integer
            evaluates batches of that size with `jax.vmap`.
        observations_are_exact_constraints: Whether observations directly fix
            state values. If `True`, their density is not added.

    Returns:
        Array: Joint log density, retaining any distribution batch axes.

    Raises:
        ValueError: If `state_path_times` is empty or missing-observation inputs
            are inconsistent.
        eqx.EquinoxRuntimeError: If a required observation or control time is
            absent from its source time array.
        NotImplementedError: If the selected missing-observation strategy is
            unsupported by the observation distribution.
    """
    state_path_times = jnp.asarray(state_path_times)
    state_path = jnp.asarray(state_path)
    _raise_now_or_error_if(
        state_path_times,
        state_path_times.shape[0] < 1,
        "state_path_times must contain at least one time point.",
    )

    initial_log_prob = dynamics.initial_condition.log_prob(state_path[0])

    if isinstance(dynamics.state_evolution, DeterministicContinuousTimeStateEvolution):
        transition_log_probs = jnp.zeros((0,), dtype=initial_log_prob.dtype)
    else:
        state_ctrl_values = _control_values_at_times(
            ctrl_times, ctrl_values, state_path_times
        )
        n_transitions = max(state_path_times.shape[0] - 1, 0)
        if n_transitions == 0:
            transition_log_probs = jnp.zeros((0,), dtype=initial_log_prob.dtype)
        else:

            def _transition_at(
                i: Int[Array, ""],
            ) -> Real[Array, "*transition_log_prob_batch"]:
                x_prev = state_path[i]
                x_next = state_path[i + 1]
                u_prev = _get_val_or_None(state_ctrl_values, i)
                transition_dist = dynamics.state_evolution(
                    x=x_prev,
                    u=u_prev,
                    t_now=state_path_times[i],
                    t_next=state_path_times[i + 1],
                )
                return transition_dist.log_prob(x_next)

            transition_log_probs = lax.map(
                _transition_at,
                jnp.arange(n_transitions),
                batch_size=chunk_size,
            )

    if obs_times is None or obs_values is None:
        return initial_log_prob + jnp.sum(transition_log_probs, axis=0)

    state_at_obs_times = _gather_by_exact_time(
        state_path,
        state_path_times,
        jnp.asarray(obs_times),
        value_name="state_path",
    )
    if observations_are_exact_constraints:
        return initial_log_prob + jnp.sum(transition_log_probs, axis=0)

    obs_ctrl_values = _control_values_at_times(ctrl_times, ctrl_values, obs_times)
    observation_log_prob, _, _, _ = prepare_observation_log_prob(
        dynamics=dynamics,
        obs_values=jnp.asarray(obs_values),
        obs_times=jnp.asarray(obs_times),
        precomputed_filled_obs=obs_values_filled,
        precomputed_obs_mask=obs_mask,
        missing_observation_strategy=missing_observation_strategy,
        missing_obs_values=missing_obs_values,
        missing_obs_metadata=missing_obs_metadata,
    )

    def _observation_at(
        i: Int[Array, ""],
    ) -> Real[Array, "*observation_log_prob_batch"]:
        return observation_log_prob(
            x=state_at_obs_times[i],
            u=_get_val_or_None(obs_ctrl_values, i),
            t=jnp.asarray(obs_times)[i],
            t_idx=i,
        )

    n_observations = jnp.asarray(obs_times).shape[0]
    if n_observations == 0:
        observation_log_probs = jnp.zeros((0,), dtype=initial_log_prob.dtype)
    else:
        observation_log_probs = lax.map(
            _observation_at,
            jnp.arange(n_observations),
            batch_size=chunk_size,
        )

    return (
        initial_log_prob
        + jnp.sum(transition_log_probs, axis=0)
        + jnp.sum(observation_log_probs, axis=0)
    )


__all__ = [
    "compute_state_path_log_prob",
]
