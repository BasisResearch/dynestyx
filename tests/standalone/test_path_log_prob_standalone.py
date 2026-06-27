import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist

import dynestyx as dsx
from dynestyx import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from tests.missingness.models import DISCRETE_A, DISCRETE_Q, GAUSSIAN_R
from tests.missingness.utils import (
    manual_masked_mvn_log_prob,
    set_full_row_missing,
    set_partial_row_missing,
)


def _make_discrete_gaussian_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=DISCRETE_A, cov=DISCRETE_Q),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
    )


def _make_hmm_dynamics():
    trans = jnp.array([[0.9, 0.1], [0.2, 0.8]])
    means = jnp.array([-1.0, 1.0])
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.Categorical(probs=jnp.array([0.4, 0.6])),
        state_evolution=lambda x, u, t_now, t_next: dist.Categorical(probs=trans[x]),
        observation_model=lambda x, u, t: dist.Normal(means[x], 0.3),
    )


def test_path_log_prob_matches_manual_full_path_score():
    dynamics = _make_discrete_gaussian_dynamics()
    times = jnp.arange(6.0)
    sim = dsx.simulate(dynamics, predict_times=times, rng_key=jr.PRNGKey(0))
    states = sim.states[0]
    obs_values = sim.observations[0]

    init_lp = dynamics.initial_condition.log_prob(states[0])
    trans_lp = sum(
        dynamics.state_evolution(
            x=states[k],
            u=None,
            t_now=times[k],
            t_next=times[k + 1],
        ).log_prob(states[k + 1])
        for k in range(len(times) - 1)
    )
    obs_lp = sum(
        dynamics.observation_model(x=states[k], u=None, t=times[k]).log_prob(
            obs_values[k]
        )
        for k in range(len(times))
    )

    actual = dsx.path_log_prob(
        dynamics,
        states,
        obs_times=times,
        obs_values=obs_values,
    )
    assert jnp.allclose(actual, init_lp + trans_lp + obs_lp)


def test_path_log_prob_matches_masked_gaussian_reference():
    dynamics = _make_discrete_gaussian_dynamics()
    times = jnp.arange(6.0)
    sim = dsx.simulate(dynamics, predict_times=times, rng_key=jr.PRNGKey(1))
    states = sim.states[0]
    obs_values = sim.observations[0]
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_partial_row_missing(obs_values, 4, dim_idx=1)

    init_lp = dynamics.initial_condition.log_prob(states[0])
    trans_lp = sum(
        dynamics.state_evolution(
            x=states[k],
            u=None,
            t_now=times[k],
            t_next=times[k + 1],
        ).log_prob(states[k + 1])
        for k in range(len(times) - 1)
    )
    obs_lp = 0.0
    for k in range(len(times)):
        mask = jnp.isfinite(obs_values[k])
        safe_obs = jnp.where(mask, obs_values[k], 0.0)
        obs_lp = obs_lp + manual_masked_mvn_log_prob(
            states[k], GAUSSIAN_R, safe_obs, mask
        )

    actual = dsx.path_log_prob(
        dynamics,
        states,
        obs_times=times,
        obs_values=obs_values,
        chunk_size=2,
    )
    assert jnp.allclose(actual, init_lp + trans_lp + obs_lp)


def test_path_log_prob_supports_hmm_missing_rows():
    dynamics = _make_hmm_dynamics()
    times = jnp.arange(8.0)
    sim = dsx.simulate(dynamics, predict_times=times, rng_key=jr.PRNGKey(2))
    states = sim.states[0, :, 0].astype(jnp.int32)
    obs_values = sim.observations[0, :, 0]
    obs_values = set_full_row_missing(obs_values, 3)
    obs_values = set_full_row_missing(obs_values, 5)

    actual = dsx.path_log_prob(
        dynamics,
        states,
        obs_times=times,
        obs_values=obs_values,
        chunk_size=3,
    )
    assert jnp.isfinite(actual)
