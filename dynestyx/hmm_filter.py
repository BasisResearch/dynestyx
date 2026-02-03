import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from dynestyx.dynamical_models import DynamicalModel


def enumerate_latent_states(dynamics: DynamicalModel) -> jnp.ndarray:
    """
    Returns all possible latent states.
    """
    K = dynamics.state_dim
    return jnp.arange(K)


def hmm_log_initial_probs(
    dynamics: DynamicalModel,
    xs: jnp.ndarray,
) -> jnp.ndarray:
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
    xs: jnp.ndarray,
    t_now,
    t_next,
    u=None,
) -> jnp.ndarray:
    """
    log p(x_{t_next} = j | x_{t_now} = i, u)
    shape: (K, K)
    """

    def row(x_prev):
        dist = dynamics.state_evolution(x=x_prev, u=u, t_now=t_now, t_next=t_next)

        def col(x_next):
            return dist.log_prob(x_next)

        return jax.vmap(col)(xs)

    return jax.vmap(row)(xs)


def hmm_log_emission_probs(
    dynamics: DynamicalModel,
    xs: jnp.ndarray,
    y,
    t,
    u=None,
) -> jnp.ndarray:
    """
    log p(y_t | x_t, u_t)
    shape: (K,)
    """

    def lp(x):
        dist = dynamics.observation_model(x=x, u=u, t=t)
        lp = dist.log_prob(y)
        return jnp.sum(lp)  # critical for vector-valued observations

    return jax.vmap(lp)(xs)


def hmm_log_components(
    dynamics: DynamicalModel,
    obs_times: jnp.ndarray,  # (T,)
    obs_values: jnp.ndarray,  # (T, ...)
    ctrl_values=None,  # (T, ...) or None
):
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
    if ctrl_values is not None:
        log_emit_seq = jax.vmap(
            lambda y, t, u: hmm_log_emission_probs(dynamics, xs, y, t, u=u)
        )(obs_values, obs_times, ctrl_values)
    else:
        log_emit_seq = jax.vmap(
            lambda y, t: hmm_log_emission_probs(dynamics, xs, y, t, u=None)
        )(obs_values, obs_times)

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
    log_pi: jnp.ndarray,  # (K,)
    log_A_seq: jnp.ndarray,  # (T-1, K, K)
    log_emit_seq: jnp.ndarray,  # (T, K)
):
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

        log_Z = logsumexp(log_alpha, axis=-1)
        log_filt = log_alpha - log_Z

        return (log_filt, loglik + log_Z), log_filt

    # t = 0
    log_alpha0 = log_pi + log_emit_seq[0]
    log_Z0 = logsumexp(log_alpha0, axis=-1)
    log_filt0 = log_alpha0 - log_Z0

    # t = 1..T-1
    # Use normalized filtered state (log_filt0) in carry for numerical stability
    (_, loglik), log_filt_rest = lax.scan(
        step,
        (log_filt0, log_Z0),
        (log_A_seq, log_emit_seq[1:]),
    )

    log_filt_seq = jnp.vstack([log_filt0[None, :], log_filt_rest])

    return loglik, log_filt_seq
