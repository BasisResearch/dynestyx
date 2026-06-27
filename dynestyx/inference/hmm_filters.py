"""HMM filter: exact forward filtering for discrete-state models."""

from typing import cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from jax import lax
from jax.scipy.special import logsumexp
from jaxtyping import Array, Bool, Float, Int, Real, Shaped

from dynestyx.inference.filter_configs import HMMConfig
from dynestyx.models import DynamicalModel
from dynestyx.models.core import DiscreteStateTransition
from dynestyx.observation_missingness import (
    masked_observation_log_prob,
    prepare_observation_views,
    probe_observation_distribution_contract,
    summarize_observation_mask,
)


def enumerate_latent_states(dynamics: DynamicalModel) -> Int[Array, " n_states"]:
    """Return all possible latent states."""
    return jnp.arange(dynamics.state_dim)


def hmm_log_initial_probs(
    dynamics: DynamicalModel,
    xs: Int[Array, " n_states"],
) -> Float[Array, " n_states"]:
    """Compute log p(x_0) for all latent states."""
    init_dist = dynamics.initial_condition
    return jax.vmap(init_dist.log_prob)(xs)


def hmm_log_transition_matrix(
    dynamics: DynamicalModel,
    xs: Int[Array, " n_states"],
    t_now: float | int | Real[Array, ""],
    t_next: float | int | Real[Array, ""],
    u=None,
) -> Float[Array, "n_states n_states"]:
    """Compute log p(x_t = j | x_{t-1} = i, u_t) for all i, j."""
    state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)

    def row(x_prev):
        transition_dist = state_transition(x=x_prev, u=u, t_now=t_now, t_next=t_next)
        return jax.vmap(transition_dist.log_prob)(xs)

    return jax.vmap(row)(xs)


def hmm_log_emission_probs_masked(
    dynamics: DynamicalModel,
    xs: Int[Array, " n_states"],
    y: Shaped[Array, " observation_dim"],
    obs_mask: Bool[Array, " observation_dim"],
    row_has_any_observed: Bool[Array, ""],
    t: float | int | Real[Array, ""],
    observation_dim: int,
    has_partial_missing: bool,
    expected_mode,
    expected_event_shape,
    u=None,
) -> Float[Array, " n_states"]:
    """Compute log p(y_t, observed entries | x_t, u_t) for each latent state."""

    def log_prob(x):
        obs_dist = dynamics.observation_model(x=x, u=u, t=t)
        return masked_observation_log_prob(
            obs_dist,
            y=y,
            obs_mask=obs_mask,
            row_has_any_observed=row_has_any_observed,
            observation_dim=observation_dim,
            has_partial_missing=has_partial_missing,
            expected_mode=expected_mode,
            expected_event_shape=expected_event_shape,
        )

    return jax.vmap(log_prob)(xs)


def hmm_log_components(
    dynamics: DynamicalModel,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"],
    _obs_values_filled: Array | None = None,
    _obs_mask: Array | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate obs_time control_dim"] | None = None,
) -> tuple[
    Float[Array, " n_states"],
    Float[Array, "*plate time_minus_1 n_states n_states"],
    Float[Array, "*plate time n_states"],
]:
    """Compute log-initial, transition, and emission terms for the HMM filter."""
    xs = enumerate_latent_states(dynamics)
    log_pi = hmm_log_initial_probs(dynamics, xs)

    if _obs_values_filled is None or _obs_mask is None:
        _obs_values_filled, _obs_mask, _ = prepare_observation_views(
            dynamics, obs_values
        )
    if _obs_values_filled is None or _obs_mask is None:
        raise ValueError("HMM observation scoring expects prepared observations.")

    obs_values_filled = (
        _obs_values_filled[:, None] if obs_values.ndim == 1 else _obs_values_filled
    )
    obs_mask = _obs_mask[:, None] if obs_values.ndim == 1 else _obs_mask
    (
        row_has_any_observed,
        _has_missing,
        has_partial_missing,
        _has_fully_missing_rows,
        observation_dim,
    ) = summarize_observation_mask(obs_mask)
    expected_mode, expected_event_shape = probe_observation_distribution_contract(
        dynamics,
        observation_dim=observation_dim,
        has_partial_missing=has_partial_missing,
    )

    if ctrl_values is not None:
        log_emit_seq = jax.vmap(
            lambda y, obs_mask_t, row_has_any, t, u: hmm_log_emission_probs_masked(
                dynamics,
                xs,
                y,
                obs_mask_t,
                row_has_any,
                t,
                observation_dim,
                has_partial_missing,
                expected_mode,
                expected_event_shape,
                u=u,
            )
        )(
            obs_values_filled,
            obs_mask,
            row_has_any_observed,
            obs_times,
            ctrl_values,
        )
    else:
        log_emit_seq = jax.vmap(
            lambda y, obs_mask_t, row_has_any, t: hmm_log_emission_probs_masked(
                dynamics,
                xs,
                y,
                obs_mask_t,
                row_has_any,
                t,
                observation_dim,
                has_partial_missing,
                expected_mode,
                expected_event_shape,
                u=None,
            )
        )(obs_values_filled, obs_mask, row_has_any_observed, obs_times)

    if ctrl_values is not None:
        log_A_seq = jax.vmap(
            lambda t_now, t_next, u_now: hmm_log_transition_matrix(
                dynamics, xs, t_now, t_next, u=u_now
            )
        )(obs_times[:-1], obs_times[1:], ctrl_values[:-1])
    else:
        log_A_seq = jax.vmap(
            lambda t_now, t_next: hmm_log_transition_matrix(
                dynamics, xs, t_now, t_next, u=None
            )
        )(obs_times[:-1], obs_times[1:])

    return log_pi, log_A_seq, log_emit_seq


def hmm_filter(
    log_pi: Float[Array, " n_states"],
    log_A_seq: Float[Array, "*plate time_minus_1 n_states n_states"],
    log_emit_seq: Float[Array, "*plate time n_states"],
) -> tuple[Shaped[Array, ""], Float[Array, "*plate time n_states"]]:
    """Run exact forward filtering and return log-likelihood and log posteriors."""

    def step(carry, inputs):
        log_filt_prev, loglik = carry
        log_A_t, log_emit_t = inputs

        log_pred = logsumexp(log_filt_prev[:, None] + log_A_t, axis=0)
        log_alpha = log_emit_t + log_pred
        log_filt = jax.nn.log_softmax(log_alpha, axis=-1)
        log_Z = logsumexp(log_alpha, axis=-1)

        return (log_filt, loglik + log_Z), log_filt

    log_alpha0 = log_pi + log_emit_seq[0]
    log_Z0 = logsumexp(log_alpha0, axis=-1)
    log_filt0 = jax.nn.log_softmax(log_alpha0, axis=-1)

    (_, loglik), log_filt_rest = lax.scan(
        step,
        (log_filt0, log_Z0),
        (log_A_seq, log_emit_seq[1:]),
    )
    log_filt_seq = jnp.vstack([log_filt0[None, :], log_filt_rest])
    return loglik, log_filt_seq


def compute_hmm_filter(
    dynamics: DynamicalModel,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"],
    _obs_values_filled: Array | None = None,
    _obs_mask: Array | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate obs_time control_dim"] | None = None,
) -> tuple[Shaped[Array, ""], Float[Array, "*plate time n_states"]]:
    """Pure-JAX HMM filter computation with missing-observation support."""
    log_pi, log_A_seq, log_emit_seq = hmm_log_components(
        dynamics,
        obs_times,
        obs_values,
        _obs_values_filled=_obs_values_filled,
        _obs_mask=_obs_mask,
        ctrl_values=ctrl_values,
    )
    return hmm_filter(log_pi, log_A_seq, log_emit_seq)


def _filter_hmm(
    name: str,
    dynamics: DynamicalModel,
    filter_config: HMMConfig,
    *,
    obs_times: Real[Array, "*obs_time_plate obs_time"],
    obs_values: Real[Array, "*obs_value_plate obs_time observation_dim"]
    | Real[Array, "*obs_value_plate obs_time"],
    _obs_values_filled: Array | None = None,
    _obs_mask: Array | None = None,
    ctrl_times: Real[Array, "*ctrl_time_plate ctrl_time"] | None = None,
    ctrl_values: Real[Array, "*ctrl_value_plate obs_time control_dim"] | None = None,
    **kwargs,
) -> tuple[jax.Array, Float[Array, "*plate time n_states"], list[dist.Distribution]]:
    """Return HMM filter outputs for deferred NumPyro site registration."""
    del name, filter_config, ctrl_times, kwargs

    loglik, log_filt_seq = compute_hmm_filter(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        _obs_values_filled=_obs_values_filled,
        _obs_mask=_obs_mask,
        ctrl_values=ctrl_values,
    )

    filtered_dists = [
        dist.Categorical(probs=jnp.exp(log_filt_seq[i]))
        for i in range(log_filt_seq.shape[0])
    ]
    return loglik, log_filt_seq, filtered_dists
