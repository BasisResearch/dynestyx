"""HMM filter: exact forward filtering for discrete-state models."""

from typing import cast

import jax
import jax.numpy as jnp
import numpyro
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
from dynestyx.utils import _should_record_field


def enumerate_latent_states(dynamics: DynamicalModel) -> Int[Array, " n_states"]:
    """
    Returns all possible latent states.
    """
    K = dynamics.state_dim
    return jnp.arange(K)


def hmm_log_initial_probs(
    dynamics: DynamicalModel,
    xs: Int[Array, " n_states"],
) -> Float[Array, " n_states"]:
    """
    log p(x_0)
    shape: (K,)
    """
    init_dist = dynamics.initial_condition

    def lp(x):
        return init_dist.log_prob(x)

    return jax.vmap(lp)(xs)


def hmm_log_transition_matrix(
    dynamics: DynamicalModel,
    xs: Int[Array, " n_states"],
    t_now: float | int | Real[Array, ""],
    t_next: float | int | Real[Array, ""],
    u=None,
) -> Float[Array, "n_states n_states"]:
    """
    log p(x_{t_next} = j | x_{t_now} = i, u)
    shape: (K, K)
    """
    state_transition = cast(DiscreteStateTransition, dynamics.state_evolution)

    def row(x_prev):
        dist = state_transition(x=x_prev, u=u, t_now=t_now, t_next=t_next)

        def col(x_next):
            return dist.log_prob(x_next)

        return jax.vmap(col)(xs)

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
    """log p(y_t, observed entries | x_t, u_t) for each latent state."""

    def lp(x):
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

    return jax.vmap(lp)(xs)


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
    """
    Returns:
      log_pi        : (K,)
      log_A_seq     : (T-1, K, K)
      log_emit_seq  : (T, K)
    """

    xs = enumerate_latent_states(dynamics)

    # Initial distribution
    log_pi = hmm_log_initial_probs(dynamics, xs)

    # Emissions
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
        )(
            obs_values_filled,
            obs_mask,
            row_has_any_observed,
            obs_times,
        )

    # Transitions
    # Note: Controls affect state evolution, use u_now for transitions from t_now to t_next
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
    """
    Exact HMM filtering.

    Returns:
      loglik       : scalar
      log_filt_seq : (T, K)  log p(x_t | y_{1:t})
    """

    def step(carry, inputs):
        log_filt_prev, loglik = carry
        log_A_t, log_emit_t = inputs

        # Predict: p(x_t | y_{1:t-1}) = sum_{x_{t-1}} p(x_t | x_{t-1}) p(x_{t-1} | y_{1:t-1})
        log_pred = logsumexp(
            log_filt_prev[:, None] + log_A_t,
            axis=0,
        )

        # Update: p(x_t | y_{1:t}) \propto p(y_t | x_t) p(x_t | y_{1:t-1})
        log_alpha = log_emit_t + log_pred
        log_filt = jax.nn.log_softmax(log_alpha, axis=-1)
        log_Z = logsumexp(log_alpha, axis=-1)

        return (log_filt, loglik + log_Z), log_filt

    # t = 0
    log_alpha0 = log_pi + log_emit_seq[0]
    log_Z0 = logsumexp(log_alpha0, axis=-1)
    log_filt0 = jax.nn.log_softmax(log_alpha0, axis=-1)

    # t = 1..T-1
    # Use normalized filtered state (log_filt0) in carry for numerical stability
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
    """Pure-JAX HMM filter computation (no numpyro side-effects).

    Returns:
        tuple: (loglik, log_filt_seq) where loglik is the marginal log-likelihood
        and log_filt_seq is (T, K) log-filtered state probabilities.
    """
    log_pi, log_A_seq, log_emit_seq = hmm_log_components(
        dynamics,
        obs_times,
        obs_values,
        _obs_values_filled=_obs_values_filled,
        _obs_mask=_obs_mask,
        ctrl_values=ctrl_values,
    )

    return hmm_filter(
        log_pi,
        log_A_seq,
        log_emit_seq,
    )


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
) -> list[dist.Distribution]:
    """Exact HMM marginal likelihood via forward filtering.

    Args:
        name: Name of the factor.
        dynamics: Dynamical model (HMM with finite discrete state space).
        filter_config: HMMConfig with record_filtered, record_log_filtered, record_max_elems.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).

    Returns:
        List of Categorical distributions p(x_t | y_{1:t}) at each obs time,
        for use with Filter + DiscreteTimeSimulator rollout.
    """
    loglik, log_filt_seq = compute_hmm_filter(
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        _obs_values_filled=_obs_values_filled,
        _obs_mask=_obs_mask,
        ctrl_values=ctrl_values,
    )

    numpyro.factor(f"{name}_marginal_log_likelihood", loglik)
    numpyro.deterministic(f"{name}_marginal_loglik", loglik)

    record_max_elems = filter_config.record_max_elems

    if _should_record_field(
        filter_config.record_log_filtered, log_filt_seq.shape, record_max_elems
    ):
        numpyro.deterministic(
            f"{name}_log_filtered_states",
            log_filt_seq,  # (T, K)
        )

    if _should_record_field(
        filter_config.record_filtered, log_filt_seq.shape, record_max_elems
    ):
        numpyro.deterministic(
            f"{name}_filtered_states",
            jnp.exp(log_filt_seq),  # (T, K)
        )

    # Return filtered distributions for Filter + DiscreteTimeSimulator rollout
    return [
        dist.Categorical(probs=jnp.exp(log_filt_seq[i]))
        for i in range(log_filt_seq.shape[0])
    ]
