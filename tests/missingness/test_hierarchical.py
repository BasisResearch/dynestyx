import jax.numpy as jnp
import jax.random as jr
from numpyro.handlers import condition, seed, trace

from dynestyx import DiscreteTimeSimulator, ODESimulator
from tests.missingness.models import (
    GAUSSIAN_R,
    plated_discrete_linear_gaussian_model,
    plated_ode_linear_gaussian_model,
)
from tests.missingness.utils import (
    latent_conditioning_data,
    manual_masked_mvn_log_prob,
    observation_log_probs,
    set_full_row_missing,
    set_partial_row_missing,
)


def _run_plated_discrete_trace(model, *, times, obs_values=None, M=2):
    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model(
                obs_times=times if obs_values is not None else None,
                obs_values=obs_values,
                predict_times=None if obs_values is not None else times,
                M=M,
            )
    return tr


def _run_plated_ode_trace(model, *, times, obs_values=None, M=2):
    with ODESimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model(
                obs_times=times if obs_values is not None else None,
                obs_values=obs_values,
                predict_times=None if obs_values is not None else times,
                M=M,
            )
    return tr


def test_hierarchical_discrete_missingness_preserves_shapes_and_member_local_factors():
    M = 2
    times = jnp.arange(4.0)
    forward = _run_plated_discrete_trace(
        plated_discrete_linear_gaussian_model, times=times, M=M
    )
    obs_values = forward["f_observations"]["value"][:, 0]
    latent_data = latent_conditioning_data(forward)

    obs_values = set_full_row_missing(obs_values, 1, member_idx=0)
    obs_values = set_partial_row_missing(obs_values, 2, dim_idx=1, member_idx=1)

    conditioned_model = condition(
        plated_discrete_linear_gaussian_model, data=latent_data
    )
    conditioned = _run_plated_discrete_trace(
        conditioned_model, times=times, obs_values=obs_values, M=M
    )

    states = conditioned["f_states"]["value"]
    observations = conditioned["f_observations"]["value"]
    assert states.shape == (M, 1, len(times), 2)
    assert observations.shape == (M, 1, len(times), 2)
    assert jnp.array_equal(jnp.isnan(observations[:, 0]), jnp.isnan(obs_values))

    member1_state = states[1, 0, 2]
    member0_log_probs = observation_log_probs(conditioned, prefix="f_p0_y")
    assert jnp.allclose(member0_log_probs[1], 0.0)

    member1_mask = jnp.isfinite(obs_values[1, 2])
    member1_safe_obs = jnp.where(member1_mask, obs_values[1, 2], 0.0)
    expected_member1 = manual_masked_mvn_log_prob(
        member1_state, GAUSSIAN_R, member1_safe_obs, member1_mask
    )
    actual_member1 = observation_log_probs(conditioned, prefix="f_p1_y")[2]
    assert jnp.allclose(actual_member1, expected_member1)


def test_hierarchical_ode_missingness_preserves_shapes_and_member_local_factors():
    M = 2
    times = jnp.linspace(0.0, 0.4, 5)
    forward = _run_plated_ode_trace(plated_ode_linear_gaussian_model, times=times, M=M)
    obs_values = forward["f_observations"]["value"][:, 0]
    latent_data = latent_conditioning_data(forward)

    obs_values = set_full_row_missing(obs_values, 1, member_idx=0)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0, member_idx=1)

    conditioned_model = condition(plated_ode_linear_gaussian_model, data=latent_data)
    conditioned = _run_plated_ode_trace(
        conditioned_model, times=times, obs_values=obs_values, M=M
    )

    states = conditioned["f_states"]["value"]
    observations = conditioned["f_observations"]["value"]
    assert states.shape == (M, 1, len(times), 2)
    assert observations.shape == (M, 1, len(times), 2)
    assert jnp.array_equal(jnp.isnan(observations[:, 0]), jnp.isnan(obs_values))

    member0_log_probs = observation_log_probs(conditioned, prefix="f_p0_y")
    assert jnp.allclose(member0_log_probs[1], 0.0)

    member1_state = states[1, 0, 3]
    member1_mask = jnp.isfinite(obs_values[1, 3])
    member1_safe_obs = jnp.where(member1_mask, obs_values[1, 3], 0.0)
    expected_member1 = manual_masked_mvn_log_prob(
        member1_state, GAUSSIAN_R, member1_safe_obs, member1_mask
    )
    actual_member1 = observation_log_probs(conditioned, prefix="f_p1_y")[3]
    assert jnp.allclose(actual_member1, expected_member1)
