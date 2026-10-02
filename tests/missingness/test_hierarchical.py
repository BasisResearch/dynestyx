import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from numpyro.handlers import condition, seed, trace

from dynestyx import DiscreteTimeSimulator, LatentPathBuilder, ODESimulator
from tests.missingness.models import (
    DISCRETE_A,
    DISCRETE_Q,
    GAUSSIAN_R,
    plated_discrete_linear_gaussian_model,
    plated_ode_linear_gaussian_model,
)
from tests.missingness.utils import (
    manual_masked_mvn_log_prob,
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


def _run_plated_discrete_latent_trace(
    model, *, times, obs_values, conditioned_data, M=2
):
    with LatentPathBuilder():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            condition(model, data=conditioned_data)(
                obs_times=times,
                obs_values=obs_values,
                M=M,
            )
    return tr


def _run_plated_ode_latent_trace(model, *, times, obs_values, conditioned_data, M=2):
    with LatentPathBuilder():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            condition(model, data=conditioned_data)(
                obs_times=times,
                obs_values=obs_values,
                M=M,
            )
    return tr


def _discrete_condition_data_from_forward(trace):
    states = jnp.asarray(trace["f_states"]["value"])[:, 0]
    return {
        f"f_p{member_idx}_state_path_params": states[member_idx]
        for member_idx in range(states.shape[0])
    }


def _ode_condition_data_from_forward(trace):
    states = jnp.asarray(trace["f_states"]["value"])[:, 0]
    return {
        f"f_p{member_idx}_state_path_params": states[member_idx, 0]
        for member_idx in range(states.shape[0])
    }


def _discrete_state_log_prob(state_path, times, *, member_idx, n_members):
    alpha = jnp.linspace(0.65, 0.78, n_members)[member_idx]
    member_A = DISCRETE_A.at[0, 0].set(alpha)
    total = jnp.asarray(0.0)
    for idx in range(len(times) - 1):
        x_prev = state_path[idx]
        x_next = state_path[idx + 1]
        total = total + dist.MultivariateNormal(
            loc=member_A @ x_prev,
            covariance_matrix=DISCRETE_Q,
        ).log_prob(x_next)
    total = total + dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=0.5 * jnp.eye(2),
    ).log_prob(state_path[0])
    return total


def _ode_states_at_obs_times(conditioned, prefix, obs_times):
    state_path = jnp.asarray(conditioned[f"{prefix}_state_path"]["value"])
    state_path_times = jnp.asarray(conditioned[f"{prefix}_state_path_times"]["value"])
    obs_indices = jnp.searchsorted(
        state_path_times, jnp.asarray(obs_times), side="left"
    )
    return state_path[obs_indices]


def test_hierarchical_discrete_missingness_preserves_shapes_and_member_local_factors():
    M = 2
    times = jnp.arange(4.0)
    forward = _run_plated_discrete_trace(
        plated_discrete_linear_gaussian_model, times=times, M=M
    )
    obs_values = forward["f_observations"]["value"][:, 0]
    conditioned_data = _discrete_condition_data_from_forward(forward)

    obs_values = set_full_row_missing(obs_values, 1, member_idx=0)
    obs_values = set_partial_row_missing(obs_values, 2, dim_idx=1, member_idx=1)

    conditioned = _run_plated_discrete_latent_trace(
        plated_discrete_linear_gaussian_model,
        times=times,
        obs_values=obs_values,
        conditioned_data=conditioned_data,
        M=M,
    )

    member0_states = conditioned["f_p0_state_path"]["value"]
    member1_states = conditioned["f_p1_state_path"]["value"]
    assert member0_states.shape == (len(times), 2)
    assert member1_states.shape == (len(times), 2)

    member0_obs_lp = conditioned["f_p0_joint_log_prob"][
        "value"
    ] - _discrete_state_log_prob(
        member0_states,
        times,
        member_idx=0,
        n_members=M,
    )
    member1_obs_lp = conditioned["f_p1_joint_log_prob"][
        "value"
    ] - _discrete_state_log_prob(
        member1_states,
        times,
        member_idx=1,
        n_members=M,
    )

    expected_member0 = jnp.sum(
        jnp.stack(
            [
                manual_masked_mvn_log_prob(
                    member0_states[k],
                    GAUSSIAN_R,
                    jnp.where(jnp.isfinite(obs_values[0, k]), obs_values[0, k], 0.0),
                    jnp.isfinite(obs_values[0, k]),
                )
                for k in range(len(times))
            ]
        )
    )
    expected_member1 = jnp.sum(
        jnp.stack(
            [
                manual_masked_mvn_log_prob(
                    member1_states[k],
                    GAUSSIAN_R,
                    jnp.where(jnp.isfinite(obs_values[1, k]), obs_values[1, k], 0.0),
                    jnp.isfinite(obs_values[1, k]),
                )
                for k in range(len(times))
            ]
        )
    )
    assert jnp.allclose(member0_obs_lp, expected_member0)
    assert jnp.allclose(member1_obs_lp, expected_member1)


def test_hierarchical_ode_missingness_preserves_shapes_and_member_local_factors():
    M = 2
    times = jnp.linspace(0.0, 0.4, 5)
    forward = _run_plated_ode_trace(plated_ode_linear_gaussian_model, times=times, M=M)
    obs_values = forward["f_observations"]["value"][:, 0]
    conditioned_data = _ode_condition_data_from_forward(forward)

    obs_values = set_full_row_missing(obs_values, 1, member_idx=0)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0, member_idx=1)

    conditioned = _run_plated_ode_latent_trace(
        plated_ode_linear_gaussian_model,
        times=times,
        obs_values=obs_values,
        conditioned_data=conditioned_data,
        M=M,
    )

    member0_state_path = conditioned["f_p0_state_path"]["value"]
    member1_state_path = conditioned["f_p1_state_path"]["value"]
    member0_states = _ode_states_at_obs_times(conditioned, "f_p0", times)
    member1_states = _ode_states_at_obs_times(conditioned, "f_p1", times)
    member0_obs_lp = conditioned["f_p0_joint_log_prob"][
        "value"
    ] - dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=0.5 * jnp.eye(2),
    ).log_prob(member0_state_path[0])
    member1_obs_lp = conditioned["f_p1_joint_log_prob"][
        "value"
    ] - dist.MultivariateNormal(
        loc=jnp.zeros(2),
        covariance_matrix=0.5 * jnp.eye(2),
    ).log_prob(member1_state_path[0])

    expected_member0 = jnp.sum(
        jnp.stack(
            [
                manual_masked_mvn_log_prob(
                    member0_states[k],
                    GAUSSIAN_R,
                    jnp.where(jnp.isfinite(obs_values[0, k]), obs_values[0, k], 0.0),
                    jnp.isfinite(obs_values[0, k]),
                )
                for k in range(len(times))
            ]
        )
    )
    expected_member1 = jnp.sum(
        jnp.stack(
            [
                manual_masked_mvn_log_prob(
                    member1_states[k],
                    GAUSSIAN_R,
                    jnp.where(jnp.isfinite(obs_values[1, k]), obs_values[1, k], 0.0),
                    jnp.isfinite(obs_values[1, k]),
                )
                for k in range(len(times))
            ]
        )
    )
    assert jnp.allclose(member0_obs_lp, expected_member0)
    assert jnp.allclose(member1_obs_lp, expected_member1)
