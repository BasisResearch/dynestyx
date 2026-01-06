import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import logsumexp
from dsx.dynamical_models import DynamicalModel

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
    xs_prev: jnp.ndarray,
    xs_next: jnp.ndarray,
    t,
) -> jnp.ndarray:
    """
    log p(x_t = j | x_{t-1} = i)
    shape: (K, K)
    """

    def row(x_prev):
        dist = dynamics.state_evolution(x=x_prev, u=None, t=t)

        def col(x_next):
            return dist.log_prob(x_next)

        return jax.vmap(col)(xs_next)

    return jax.vmap(row)(xs_prev)

def hmm_log_emission_probs(
    dynamics: DynamicalModel,
    xs: jnp.ndarray,
    y,
    t,
) -> jnp.ndarray:
    """
    log p(y_t | x_t)
    shape: (K,)
    """

    def lp(x):
        dist = dynamics.observation_model(x=x, u=None, t=t)
        return dist.log_prob(y)

    return jax.vmap(lp)(xs)

def hmm_log_components(
    dynamics: DynamicalModel,
    obs_times: jnp.ndarray,   # (T,)
    obs_values: jnp.ndarray,  # (T, ...)
):
    """
    Returns:
      log_pi        : (K,)
      log_A_seq     : (T-1, K, K)
      log_emit_seq  : (T, K)
    """

    xs = enumerate_latent_states(dynamics)
    T = obs_times.shape[0]

    # Initial distribution
    log_pi = hmm_log_initial_probs(dynamics, xs)

    # Emissions
    log_emit_seq = jax.vmap(
        lambda y, t: hmm_log_emission_probs(dynamics, xs, y, t)
    )(obs_values, obs_times)

    # Transitions
    if T > 1:
        log_A_seq = jax.vmap(
            lambda t: hmm_log_transition_matrix(dynamics, xs, xs, t)
        )(obs_times[:-1])
    else:
        log_A_seq = jnp.zeros((0, xs.shape[0], xs.shape[0]))

    return log_pi, log_A_seq, log_emit_seq

def hmm_filter(
    log_pi: jnp.ndarray,       # (K,)
    log_A_seq: jnp.ndarray,    # (T-1, K, K)
    log_emit_seq: jnp.ndarray, # (T, K)
):
    """
    Exact HMM filtering.

    Returns:
      loglik       : scalar
      log_filt_seq : (T, K)  log p(x_t | y_{1:t})
    """

    def step(carry, inputs):
        log_alpha_prev, loglik = carry
        log_A_t, log_emit_t = inputs

        log_alpha = log_emit_t + logsumexp(
            log_alpha_prev[:, None] + log_A_t,
            axis=0,
        )

        log_Z = logsumexp(log_alpha)
        log_filt = log_alpha - log_Z

        return (log_alpha, loglik + log_Z), log_filt

    # t = 0
    log_alpha0 = log_pi + log_emit_seq[0]
    log_Z0 = logsumexp(log_alpha0)
    log_filt0 = log_alpha0 - log_Z0

    # t = 1..T-1
    (_, loglik), log_filt_rest = lax.scan(
        step,
        (log_alpha0, log_Z0),
        (log_A_seq, log_emit_seq[1:]),
    )

    log_filt_seq = jnp.vstack(
        [log_filt0[None, :], log_filt_rest]
    )

    return loglik, log_filt_seq
