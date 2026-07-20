import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import condition, seed, trace
from numpyro.infer import MCMC, NUTS, Predictive

import dynestyx as dsx
from dynestyx import (
    DiscreteTimeSimulator,
    DynamicalModel,
    GaussianObservation,
    LatentPathBuilder,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)
from tests.missingness.models import (
    DISCRETE_A,
    DISCRETE_Q,
    GAUSSIAN_R,
    INDEPENDENT_SCALE,
    _independent_observation_mean,
    _nonlinear_observation_mean,
    discrete_dirac_model,
    discrete_independent_normal_model,
    discrete_linear_gaussian_model,
    discrete_nonlinear_gaussian_model,
    sampled_discrete_linear_gaussian_model,
)
from tests.missingness.utils import (
    manual_masked_independent_normal_log_prob,
    manual_masked_mvn_log_prob,
    set_full_row_missing,
    set_partial_row_missing,
)


def _run_discrete_trace(model, *, obs_times=None, obs_values=None, predict_times=None):
    with DiscreteTimeSimulator():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model(
                obs_times=obs_times,
                obs_values=obs_values,
                predict_times=predict_times,
            )
    return tr


def _run_discrete_latent_trace(model, *, obs_times, obs_values, conditioned_data=None):
    model_to_run = (
        model if conditioned_data is None else condition(model, data=conditioned_data)
    )
    with LatentPathBuilder():
        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            model_to_run(
                obs_times=obs_times,
                obs_values=obs_values,
            )
    return tr


def _discrete_state_path_params_from_forward(trace):
    return jnp.asarray(trace["f_states"]["value"])[0]


def _discrete_state_log_prob(dynamics, state_path, times):
    total = dynamics.initial_condition.log_prob(state_path[0])
    for idx in range(len(times) - 1):
        total = total + dynamics.state_evolution(
            state_path[idx],
            None,
            times[idx],
            times[idx + 1],
        ).log_prob(state_path[idx + 1])
    return total


def _make_discrete_linear_gaussian_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=DISCRETE_A, cov=DISCRETE_Q),
        observation_model=LinearGaussianObservation(H=jnp.eye(2), R=GAUSSIAN_R),
    )


def _make_discrete_nonlinear_gaussian_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=DISCRETE_A, cov=DISCRETE_Q),
        observation_model=GaussianObservation(
            h=_nonlinear_observation_mean, R=GAUSSIAN_R
        ),
    )


def _make_discrete_independent_normal_dynamics():
    return DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(
            loc=jnp.zeros(2), covariance_matrix=0.5 * jnp.eye(2)
        ),
        state_evolution=LinearGaussianStateEvolution(A=DISCRETE_A, cov=DISCRETE_Q),
        observation_model=lambda x, u, t: dist.Independent(
            dist.Normal(_independent_observation_mean(x, u, t), INDEPENDENT_SCALE), 1
        ),
    )


def _correlated_student_t_model(
    alpha=None,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    alpha = numpyro.sample("alpha", dist.Uniform(-0.7, 0.7), obs=alpha)
    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=LinearGaussianStateEvolution(
            A=jnp.array([[alpha, 0.2], [-0.1, 0.8]]),
            cov=0.05 * jnp.eye(2),
        ),
        observation_model=lambda x, u, t: dist.MultivariateStudentT(
            df=5.0,
            loc=x,
            scale_tril=jnp.array([[0.4, 0.0], [0.15, 0.5]]),
        ),
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def _scalar_categorical_hmm_like_model(
    A=None,
    obs_times=None,
    obs_values=None,
    predict_times=None,
):
    A = numpyro.sample(
        "A",
        dist.Dirichlet(jnp.ones(2)).expand([2]).to_event(1),
        obs=A,
    )

    def state_evolution(x, u, t_now, t_next):
        return dist.Categorical(probs=A[x])

    def observation_model(x, u, t):
        probs = jnp.array(
            [
                [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
                [1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 10, 1 / 2],
            ]
        )
        return dist.Categorical(probs=probs[x])

    dynamics = DynamicalModel(
        control_dim=0,
        initial_condition=dist.Categorical(probs=jnp.ones(2) / 2),
        state_evolution=state_evolution,
        observation_model=observation_model,
    )
    return dsx.sample(
        "f",
        dynamics,
        obs_times=obs_times,
        obs_values=obs_values,
        predict_times=predict_times,
    )


def test_discrete_no_missing_conditioning_uses_log_prob_path():
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(discrete_linear_gaussian_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    state_path_params = _discrete_state_path_params_from_forward(forward)
    conditioned = _run_discrete_latent_trace(
        discrete_linear_gaussian_model,
        obs_times=times,
        obs_values=obs_values,
        conditioned_data={"f_state_path_params": state_path_params},
    )

    states = conditioned["f_state_path"]["value"]
    expected_observation_log_prob = jnp.sum(
        jnp.stack(
            [
                manual_masked_mvn_log_prob(
                    states[k],
                    GAUSSIAN_R,
                    obs_values[k],
                    jnp.ones_like(obs_values[k], dtype=bool),
                )
                for k in range(len(times))
            ]
        )
    )
    expected_joint_log_prob = (
        _discrete_state_log_prob(
            _make_discrete_linear_gaussian_dynamics(),
            states,
            times,
        )
        + expected_observation_log_prob
    )
    assert jnp.allclose(states, state_path_params)
    assert jnp.allclose(
        conditioned["f_joint_log_prob"]["value"],
        expected_joint_log_prob,
    )


@pytest.mark.parametrize(
    ("model", "mean_fn"),
    [
        (discrete_linear_gaussian_model, lambda x, t: x),
        (
            discrete_nonlinear_gaussian_model,
            lambda x, t: _nonlinear_observation_mean(x, None, t),
        ),
    ],
)
def test_discrete_gaussian_missingness_factor_values_match_manual_reference(
    model,
    mean_fn,
):
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    state_path_params = _discrete_state_path_params_from_forward(forward)
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0)

    conditioned = _run_discrete_latent_trace(
        model,
        obs_times=times,
        obs_values=obs_values,
        conditioned_data={"f_state_path_params": state_path_params},
    )

    states = conditioned["f_state_path"]["value"]
    assert states.shape == (len(times), 2)

    expected = []
    for k in range(len(times)):
        mask = jnp.isfinite(obs_values[k])
        safe_obs = jnp.where(mask, obs_values[k], 0.0)
        mu = mean_fn(states[k], times[k])
        expected.append(manual_masked_mvn_log_prob(mu, GAUSSIAN_R, safe_obs, mask))

    dynamics = (
        _make_discrete_linear_gaussian_dynamics()
        if model is discrete_linear_gaussian_model
        else _make_discrete_nonlinear_gaussian_dynamics()
    )
    actual = conditioned["f_joint_log_prob"]["value"] - _discrete_state_log_prob(
        dynamics,
        states,
        times,
    )
    assert jnp.allclose(actual, jnp.sum(jnp.stack(expected)))


def test_discrete_independent_missingness_factor_values_match_manual_reference():
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(
        discrete_independent_normal_model, predict_times=times
    )
    obs_values = forward["f_observations"]["value"][0]
    state_path_params = _discrete_state_path_params_from_forward(forward)
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_partial_row_missing(obs_values, 4, dim_idx=1)

    conditioned = _run_discrete_latent_trace(
        discrete_independent_normal_model,
        obs_times=times,
        obs_values=obs_values,
        conditioned_data={"f_state_path_params": state_path_params},
    )

    states = conditioned["f_state_path"]["value"]

    expected = []
    for k in range(len(times)):
        mask = jnp.isfinite(obs_values[k])
        safe_obs = jnp.where(mask, obs_values[k], 0.0)
        loc = _independent_observation_mean(states[k], None, times[k])
        expected.append(
            manual_masked_independent_normal_log_prob(
                loc, INDEPENDENT_SCALE, safe_obs, mask
            )
        )

    dynamics = _make_discrete_independent_normal_dynamics()
    actual = conditioned["f_joint_log_prob"]["value"] - _discrete_state_log_prob(
        dynamics,
        states,
        times,
    )
    assert jnp.allclose(actual, jnp.sum(jnp.stack(expected)))


def test_discrete_missingness_mcmc_smoke():
    times = jnp.arange(5.0)
    predictive = Predictive(
        sampled_discrete_linear_gaussian_model,
        params={"alpha": jnp.array(0.72)},
        num_samples=1,
        exclude_deterministic=False,
    )
    with DiscreteTimeSimulator():
        generated = predictive(jr.PRNGKey(1), predict_times=times)
    obs_values = generated["f_observations"][0, 0]
    obs_values = set_full_row_missing(obs_values, 1)
    obs_values = set_partial_row_missing(obs_values, 3, dim_idx=0)

    with LatentPathBuilder():
        mcmc = MCMC(
            NUTS(sampled_discrete_linear_gaussian_model),
            num_samples=1,
            num_warmup=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(2), obs_times=times, obs_values=obs_values)

    assert "alpha" in mcmc.get_samples()


def test_discrete_full_row_missing_correlated_student_t_mcmc_smoke():
    times = jnp.arange(8.0)
    with DiscreteTimeSimulator():
        generated = Predictive(
            _correlated_student_t_model,
            params={"alpha": jnp.array(0.3)},
            num_samples=1,
            exclude_deterministic=False,
        )(jr.PRNGKey(3), predict_times=times)
    obs_values = generated["f_observations"][0, 0]
    obs_values = set_full_row_missing(obs_values, 2)
    obs_values = set_full_row_missing(obs_values, 3)
    obs_values = set_full_row_missing(obs_values, 4)

    with LatentPathBuilder():
        mcmc = MCMC(
            NUTS(_correlated_student_t_model),
            num_samples=1,
            num_warmup=1,
            progress_bar=False,
        )
        mcmc.run(jr.PRNGKey(4), obs_times=times, obs_values=obs_values)

    assert "alpha" in mcmc.get_samples()


def test_discrete_categorical_conditioning_raises_clear_error():
    times = jnp.arange(6.0)
    true_A = jnp.array([[0.95, 0.05], [0.1, 0.9]])
    with DiscreteTimeSimulator():
        generated = Predictive(_scalar_categorical_hmm_like_model, num_samples=1)(
            jr.PRNGKey(5),
            A=true_A,
            predict_times=times,
        )
    obs_values = jnp.asarray(generated["f_observations"])[0, 0, :, 0]

    with pytest.raises(
        ValueError,
        match="generation-only",
    ):
        with DiscreteTimeSimulator():
            with seed(rng_seed=jr.PRNGKey(6)):
                _scalar_categorical_hmm_like_model(
                    A=true_A,
                    obs_times=times,
                    obs_values=obs_values,
                )


def test_discrete_dirac_missingness_raises_clear_error():
    times = jnp.arange(5.0)
    forward = _run_discrete_trace(discrete_dirac_model, predict_times=times)
    obs_values = forward["f_observations"]["value"][0]
    obs_values = set_full_row_missing(obs_values, 2)

    with pytest.raises(
        ValueError,
        match="generation-only",
    ):
        _run_discrete_trace(
            discrete_dirac_model, obs_times=times, obs_values=obs_values
        )
