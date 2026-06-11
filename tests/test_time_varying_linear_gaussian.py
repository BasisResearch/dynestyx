"""Tests for time-varying (callable-parameter) linear-Gaussian models.

`LinearGaussianStateEvolution` parameters may be callables of
``(t_now, t_next)`` and `LinearGaussianObservation` parameters callables of
``(t,)``. These tests pin:

- callable parameters returning constants match the constant-parameter model
  exactly (filter, smoother, simulator);
- genuinely time-varying parameters are respected (results differ from the
  time-invariant model, and match a hand-rolled reference Kalman filter);
- mixed constant/callable combinations run end-to-end through ``dsx.sample``
  + ``Filter``;
- the cd_dynamax backend rejects callable parameters with a clear TypeError.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as jsp_linalg
import numpyro
import numpyro.distributions as dist
import pytest
from numpyro.handlers import seed, trace

import dynestyx as dsx
from dynestyx.inference.filter_configs import EKFConfig, EnKFConfig, KFConfig
from dynestyx.inference.filters import Filter
from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (
    _lti_to_lgssm_params,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    dsx_to_cd_dynamax,
    dsx_to_cdlgssm_params,
    gaussian_to_nlgssm_params,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    compute_cuthbert_filter,
    compute_cuthbert_smoother,
)
from dynestyx.inference.mcmc import MCMCInference
from dynestyx.inference.mcmc_configs import NUTSConfig
from dynestyx.inference.smoother_configs import EKFSmootherConfig, KFSmootherConfig
from dynestyx.inference.smoothers import Smoother
from dynestyx.models import (
    AffineDrift,
    DynamicalModel,
    FullDiffusion,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
    StochasticContinuousTimeStateEvolution,
)
from dynestyx.simulators import DiscreteTimeSimulator

A_C = jnp.array([[-0.5, 0.4], [0.0, -0.3]])
A_CONST = jsp_linalg.expm(A_C)
Q0 = jnp.array([[0.2, 0.05], [0.05, 0.1]])
H0 = jnp.array([[1.0, 0.0], [0.5, 1.0]])
R0 = 0.05 * jnp.eye(2)
B0 = jnp.array([[0.1], [0.3]])
BIAS0 = jnp.array([0.05, -0.02])
D0 = jnp.array([[0.02], [0.01]])
OBS_BIAS0 = jnp.array([0.01, 0.02])
IRREGULAR_TIMES = jnp.array([0.0, 0.3, 1.0, 1.1, 2.5, 2.6, 4.0])


# --- Genuinely time-varying parameters (exact CT-LTI discretization style) ---


def _transition_matrix(t_now, t_next):
    return jsp_linalg.expm(A_C * (t_next - t_now))


def _transition_cov(t_now, t_next):
    return Q0 * (t_next - t_now)


def _transition_control(t_now, t_next):
    return B0 * (t_next - t_now)


def _transition_bias(t_now, t_next):
    return BIAS0 * (t_next - t_now)


def _observation_matrix(t):
    return H0 * (1.0 + 0.1 * t)


def _observation_cov(t):
    return R0 * (1.0 + 0.05 * t)


def _observation_control(t):
    return D0 * (1.0 + 0.1 * t)


def _observation_bias(t):
    return OBS_BIAS0 * (1.0 + t)


# --- Callables returning the constants (for exact-equivalence tests) ---


def _constant_transition_matrix(t_now, t_next):
    del t_now, t_next
    return A_CONST


def _constant_transition_cov(t_now, t_next):
    del t_now, t_next
    return Q0


def _constant_transition_control(t_now, t_next):
    del t_now, t_next
    return B0


def _constant_transition_bias(t_now, t_next):
    del t_now, t_next
    return BIAS0


def _constant_observation_matrix(t):
    del t
    return H0


def _constant_observation_cov(t):
    del t
    return R0


def _constant_observation_control(t):
    del t
    return D0


def _constant_observation_bias(t):
    del t
    return OBS_BIAS0


def _make_model(evo_kwargs, obs_kwargs, control_dim=0):
    return DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=LinearGaussianStateEvolution(**evo_kwargs),
        observation_model=LinearGaussianObservation(**obs_kwargs),
        control_dim=control_dim,
    )


def _make_obs_values(key=0):
    return jax.random.normal(jr.PRNGKey(key), (len(IRREGULAR_TIMES), 2))


def _make_ctrl_values():
    return jnp.sin(IRREGULAR_TIMES)[:, None]


# --- Equivalence: callable-with-constant-return matches constant fields ---


@pytest.mark.parametrize("associative", [False, True])
def test_cuthbert_kf_callable_constant_matches_constant(associative):
    obs_values = _make_obs_values()
    ctrl_values = _make_ctrl_values()
    constant_model = _make_model(
        dict(A=A_CONST, cov=Q0, B=B0, bias=BIAS0),
        dict(H=H0, R=R0, D=D0, bias=OBS_BIAS0),
        control_dim=1,
    )
    callable_model = _make_model(
        dict(
            A=_constant_transition_matrix,
            cov=_constant_transition_cov,
            B=_constant_transition_control,
            bias=_constant_transition_bias,
        ),
        dict(
            H=_constant_observation_matrix,
            R=_constant_observation_cov,
            D=_constant_observation_control,
            bias=_constant_observation_bias,
        ),
        control_dim=1,
    )

    config = KFConfig(filter_source="cuthbert", associative=associative)
    loglik_constant, states_constant = compute_cuthbert_filter(
        constant_model,
        config,
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
    )
    loglik_callable, states_callable = compute_cuthbert_filter(
        callable_model,
        config,
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
        ctrl_values=ctrl_values,
    )

    assert jnp.allclose(loglik_constant, loglik_callable, atol=1e-6)
    assert jnp.allclose(states_constant.mean, states_callable.mean, atol=1e-6)
    assert jnp.allclose(states_constant.chol_cov, states_callable.chol_cov, atol=1e-6)


def test_cuthbert_kf_smoother_callable_constant_matches_constant():
    obs_values = _make_obs_values()
    constant_model = _make_model(dict(A=A_CONST, cov=Q0), dict(H=H0, R=R0))
    callable_model = _make_model(
        dict(A=_constant_transition_matrix, cov=_constant_transition_cov),
        dict(H=_constant_observation_matrix, R=_constant_observation_cov),
    )

    config = KFSmootherConfig(filter_source="cuthbert")
    loglik_constant, smoothed_constant = compute_cuthbert_smoother(
        constant_model, config, obs_times=IRREGULAR_TIMES, obs_values=obs_values
    )
    loglik_callable, smoothed_callable = compute_cuthbert_smoother(
        callable_model, config, obs_times=IRREGULAR_TIMES, obs_values=obs_values
    )

    assert jnp.allclose(loglik_constant, loglik_callable, atol=1e-6)
    assert jnp.allclose(smoothed_constant.mean, smoothed_callable.mean, atol=1e-6)
    assert jnp.allclose(
        smoothed_constant.chol_cov, smoothed_callable.chol_cov, atol=1e-6
    )


# --- Time variation is respected ---


def test_cuthbert_kf_time_varying_differs_from_time_invariant():
    obs_values = _make_obs_values()
    # Time-invariant model frozen at the dt = 1 / t = 0 parameter values.
    frozen_model = _make_model(dict(A=A_CONST, cov=Q0), dict(H=H0, R=R0))
    time_varying_dynamics_model = _make_model(
        dict(A=_transition_matrix, cov=_transition_cov), dict(H=H0, R=R0)
    )
    time_varying_observation_model = _make_model(
        dict(A=A_CONST, cov=Q0), dict(H=_observation_matrix, R=R0)
    )

    config = KFConfig(filter_source="cuthbert")
    loglik_frozen, states_frozen = compute_cuthbert_filter(
        frozen_model, config, obs_times=IRREGULAR_TIMES, obs_values=obs_values
    )
    loglik_tv_dyn, states_tv_dyn = compute_cuthbert_filter(
        time_varying_dynamics_model,
        config,
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )
    loglik_tv_obs, _ = compute_cuthbert_filter(
        time_varying_observation_model,
        config,
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )

    assert jnp.isfinite(loglik_tv_dyn) and jnp.isfinite(loglik_tv_obs)
    assert jnp.abs(loglik_tv_dyn - loglik_frozen) > 1e-2
    assert jnp.abs(loglik_tv_obs - loglik_frozen) > 1e-2
    assert not jnp.allclose(states_tv_dyn.mean, states_frozen.mean, atol=1e-3)


def _reference_kalman_filter(times, obs_values, A_fn, Q_fn, H_fn, R_fn):
    """Textbook time-varying Kalman filter (no controls/biases).

    Convention matches dynestyx: the initial condition is the prior for the
    state at the first observation time (no predict step before the first
    update).
    """
    mean = jnp.zeros(2)
    cov = jnp.eye(2)
    loglik = 0.0
    means, covs = [], []
    for k in range(len(times)):
        if k > 0:
            A_k = A_fn(times[k - 1], times[k])
            mean = A_k @ mean
            cov = A_k @ cov @ A_k.T + Q_fn(times[k - 1], times[k])
        H_k = H_fn(times[k])
        innovation = obs_values[k] - H_k @ mean
        innovation_cov = H_k @ cov @ H_k.T + R_fn(times[k])
        gain = cov @ H_k.T @ jnp.linalg.inv(innovation_cov)
        loglik = loglik + dist.MultivariateNormal(H_k @ mean, innovation_cov).log_prob(
            obs_values[k]
        )
        mean = mean + gain @ innovation
        cov = cov - gain @ innovation_cov @ gain.T
        means.append(mean)
        covs.append(cov)
    return loglik, jnp.stack(means), jnp.stack(covs)


def test_cuthbert_kf_time_varying_matches_reference_kalman():
    obs_values = _make_obs_values()
    model = _make_model(
        dict(A=_transition_matrix, cov=_transition_cov),
        dict(H=_observation_matrix, R=_observation_cov),
    )

    loglik, states = compute_cuthbert_filter(
        model,
        KFConfig(filter_source="cuthbert"),
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )
    reference_loglik, reference_means, reference_covs = _reference_kalman_filter(
        IRREGULAR_TIMES,
        obs_values,
        _transition_matrix,
        _transition_cov,
        _observation_matrix,
        _observation_cov,
    )

    filtered_covs = states.chol_cov @ jnp.swapaxes(states.chol_cov, -1, -2)
    assert jnp.allclose(loglik, reference_loglik, atol=1e-4)
    assert jnp.allclose(states.mean, reference_means, atol=1e-4)
    assert jnp.allclose(filtered_covs, reference_covs, atol=1e-4)


def test_cuthbert_kf_time_varying_matches_cuthbert_ekf():
    # The cuthbert Taylor (EKF) path goes through state_evolution.__call__ /
    # observation_model.__call__, which are exact for linear-Gaussian models:
    # it cross-validates the per-step params_at wiring of the exact KF path.
    obs_values = _make_obs_values()
    model = _make_model(
        dict(A=_transition_matrix, cov=_transition_cov),
        dict(H=_observation_matrix, R=R0),
    )

    loglik_kf, states_kf = compute_cuthbert_filter(
        model,
        KFConfig(filter_source="cuthbert"),
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )
    loglik_ekf, states_ekf = compute_cuthbert_filter(
        model,
        EKFConfig(filter_source="cuthbert"),
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )

    assert jnp.allclose(states_kf.mean, states_ekf.mean, atol=1e-3)
    covs_kf = states_kf.chol_cov @ jnp.swapaxes(states_kf.chol_cov, -1, -2)
    covs_ekf = states_ekf.chol_cov @ jnp.swapaxes(states_ekf.chol_cov, -1, -2)
    assert jnp.allclose(covs_kf, covs_ekf, atol=1e-3)
    # The Taylor path's potential-form observation update drops the
    # log|det H| normalizer, so its marginal loglik is offset by
    # sum_k log|det H(t_k)| (invisible when det H == 1, as in every constant-H
    # fixture). The exact KF value is pinned against a brute-force joint
    # Gaussian via test_cuthbert_kf_time_varying_matches_reference_kalman.
    log_det_offset = jnp.sum(
        2.0 * jnp.log(1.0 + 0.1 * IRREGULAR_TIMES)
    )  # log|det H(t)| for H(t) = H0 * (1 + 0.1 t) with det H0 = 1
    assert jnp.allclose(loglik_ekf - loglik_kf, log_det_offset, atol=1e-2)


def test_cuthbert_kf_smoother_time_varying():
    obs_values = _make_obs_values()
    model = _make_model(
        dict(A=_transition_matrix, cov=_transition_cov),
        dict(H=_observation_matrix, R=R0),
    )

    loglik_kf, smoothed_kf = compute_cuthbert_smoother(
        model,
        KFSmootherConfig(filter_source="cuthbert"),
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )
    # The Taylor (EKF) smoother's posterior moments are exact for
    # linear-Gaussian models and evaluate the dynamics/observations through
    # __call__, independent of the KF smoother's per-step params_at wiring.
    # (Logliks are not compared: the Taylor path's normalizer is offset by
    # sum_k log|det H(t_k)|; see
    # test_cuthbert_kf_time_varying_matches_cuthbert_ekf.)
    _, smoothed_ekf = compute_cuthbert_smoother(
        model,
        EKFSmootherConfig(filter_source="cuthbert"),
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )

    assert jnp.isfinite(loglik_kf)
    assert jnp.allclose(smoothed_kf.mean, smoothed_ekf.mean, atol=1e-3)


# --- Mixed constant/callable combinations through dsx.sample + Filter ---

MIXED_CASES = {
    "callable_A": (dict(A=_transition_matrix, cov=Q0), dict(H=H0, R=R0), False),
    "callable_cov": (dict(A=A_CONST, cov=_transition_cov), dict(H=H0, R=R0), False),
    "callable_B_with_controls": (
        dict(A=A_CONST, cov=Q0, B=_transition_control),
        dict(H=H0, R=R0),
        True,
    ),
    "callable_bias": (
        dict(A=A_CONST, cov=Q0, bias=_transition_bias),
        dict(H=H0, R=R0),
        False,
    ),
    "callable_H": (dict(A=A_CONST, cov=Q0), dict(H=_observation_matrix, R=R0), False),
    "callable_R": (dict(A=A_CONST, cov=Q0), dict(H=H0, R=_observation_cov), False),
    "callable_D_with_controls": (
        dict(A=A_CONST, cov=Q0),
        dict(H=H0, R=R0, D=_observation_control),
        True,
    ),
    "callable_obs_bias": (
        dict(A=A_CONST, cov=Q0),
        dict(H=H0, R=R0, bias=_observation_bias),
        False,
    ),
    "all_callable": (
        dict(
            A=_transition_matrix,
            cov=_transition_cov,
            B=_transition_control,
            bias=_transition_bias,
        ),
        dict(
            H=_observation_matrix,
            R=_observation_cov,
            D=_observation_control,
            bias=_observation_bias,
        ),
        True,
    ),
}


@pytest.mark.parametrize("case", sorted(MIXED_CASES))
def test_cuthbert_kf_mixed_constant_callable_combinations(case):
    evo_kwargs, obs_kwargs, with_controls = MIXED_CASES[case]
    obs_values = _make_obs_values()
    sample_kwargs = dict(obs_times=IRREGULAR_TIMES, obs_values=obs_values)
    control_dim = 0
    if with_controls:
        control_dim = 1
        sample_kwargs.update(
            ctrl_times=IRREGULAR_TIMES, ctrl_values=_make_ctrl_values()
        )

    def model():
        dynamics = _make_model(evo_kwargs, obs_kwargs, control_dim=control_dim)
        dsx.sample("f", dynamics, **sample_kwargs)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
            model()

    marginal_loglik = tr["f_marginal_loglik"]["value"]
    assert jnp.ndim(marginal_loglik) == 0
    assert jnp.isfinite(marginal_loglik)
    filtered_means = tr["f_filtered_states_mean"]["value"]
    assert filtered_means.shape == (len(IRREGULAR_TIMES), 2)
    assert jnp.isfinite(filtered_means).all()


# --- cd_dynamax rejects callable parameters ---

EVO_FIELD_CALLABLES = {
    "A": _transition_matrix,
    "B": _transition_control,
    "bias": _transition_bias,
    "cov": _transition_cov,
}
OBS_FIELD_CALLABLES = {
    "H": _observation_matrix,
    "D": _observation_control,
    "bias": _observation_bias,
    "R": _observation_cov,
}


@pytest.mark.parametrize("converter", [_lti_to_lgssm_params, gaussian_to_nlgssm_params])
@pytest.mark.parametrize("field", sorted(EVO_FIELD_CALLABLES))
def test_cd_dynamax_discrete_rejects_callable_evolution_fields(converter, field):
    evo_kwargs = {"A": A_CONST, "cov": Q0, field: EVO_FIELD_CALLABLES[field]}
    control_dim = 1 if field == "B" else 0
    model = _make_model(evo_kwargs, dict(H=H0, R=R0), control_dim=control_dim)

    with pytest.raises(TypeError, match="callable") as excinfo:
        converter(model)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("converter", [_lti_to_lgssm_params, gaussian_to_nlgssm_params])
@pytest.mark.parametrize("field", sorted(OBS_FIELD_CALLABLES))
def test_cd_dynamax_discrete_rejects_callable_observation_fields(converter, field):
    obs_kwargs = {"H": H0, "R": R0, field: OBS_FIELD_CALLABLES[field]}
    control_dim = 1 if field == "D" else 0
    model = _make_model(dict(A=A_CONST, cov=Q0), obs_kwargs, control_dim=control_dim)

    with pytest.raises(TypeError, match="callable") as excinfo:
        converter(model)
    assert field in str(excinfo.value)


@pytest.mark.parametrize("converter", [dsx_to_cdlgssm_params, dsx_to_cd_dynamax])
def test_cd_dynamax_continuous_rejects_callable_observation_fields(converter):
    model = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=StochasticContinuousTimeStateEvolution(
            drift=AffineDrift(A=A_C),
            diffusion=FullDiffusion(0.3 * jnp.eye(2)),
        ),
        observation_model=LinearGaussianObservation(H=_observation_matrix, R=R0),
    )

    with pytest.raises(TypeError, match="callable") as excinfo:
        converter(model)
    assert "H" in str(excinfo.value)


def test_cd_dynamax_filter_and_smoother_handlers_reject_time_varying():
    obs_values = _make_obs_values()

    def model():
        dynamics = _make_model(
            dict(A=_transition_matrix, cov=_transition_cov), dict(H=H0, R=R0)
        )
        dsx.sample("f", dynamics, obs_times=IRREGULAR_TIMES, obs_values=obs_values)

    with pytest.raises(TypeError, match="callable"):
        with seed(rng_seed=jr.PRNGKey(0)):
            with Filter(filter_config=KFConfig(filter_source="cd_dynamax")):
                model()

    with pytest.raises(TypeError, match="callable"):
        with seed(rng_seed=jr.PRNGKey(0)):
            with Smoother(smoother_config=KFSmootherConfig(filter_source="cd_dynamax")):
                model()


# --- Simulators ---


def test_discrete_time_simulator_time_varying():
    def run_simulation(evo_kwargs, obs_kwargs):
        def model():
            dynamics = _make_model(evo_kwargs, obs_kwargs)
            dsx.sample("f", dynamics, predict_times=IRREGULAR_TIMES)

        with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
            with DiscreteTimeSimulator():
                model()
        return tr["f_states"]["value"], tr["f_observations"]["value"]

    states_constant, observations_constant = run_simulation(
        dict(A=A_CONST, cov=Q0), dict(H=H0, R=R0)
    )
    states_callable, observations_callable = run_simulation(
        dict(A=_constant_transition_matrix, cov=_constant_transition_cov),
        dict(H=_constant_observation_matrix, R=_constant_observation_cov),
    )
    assert jnp.allclose(states_constant, states_callable, atol=1e-6)
    assert jnp.allclose(observations_constant, observations_callable, atol=1e-6)

    states_tv, observations_tv = run_simulation(
        dict(A=_transition_matrix, cov=_transition_cov),
        dict(H=_observation_matrix, R=R0),
    )
    assert jnp.isfinite(states_tv).all() and jnp.isfinite(observations_tv).all()
    assert states_tv.shape == states_constant.shape
    assert not jnp.allclose(states_tv, states_constant, atol=1e-6)


# --- Other cuthbert backends pick up time variation through __call__ ---


def test_cuthbert_enkf_time_varying_smoke():
    obs_values = _make_obs_values()
    model = _make_model(
        dict(A=_transition_matrix, cov=_transition_cov),
        dict(H=_observation_matrix, R=R0),
    )

    marginal_loglik, _ = compute_cuthbert_filter(
        model,
        EnKFConfig(n_particles=16, filter_source="cuthbert"),
        key=jr.PRNGKey(3),
        obs_times=IRREGULAR_TIMES,
        obs_values=obs_values,
    )
    assert jnp.isfinite(marginal_loglik)


# --- Plates: shared callable parameters survive the batched vmap path ---


def test_plate_cuthbert_kf_with_shared_callable_cov_smoke():
    n_members = 3
    batched_A = jnp.stack([0.9 * A_CONST, A_CONST, 1.1 * A_CONST])
    obs_values = jax.random.normal(jr.PRNGKey(1), (n_members, len(IRREGULAR_TIMES), 2))

    def model():
        with dsx.plate("trajectories", n_members):
            dynamics = DynamicalModel(
                initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
                state_evolution=LinearGaussianStateEvolution(
                    A=batched_A, cov=_transition_cov
                ),
                observation_model=LinearGaussianObservation(H=H0, R=R0),
            )
            dsx.sample("f", dynamics, obs_times=IRREGULAR_TIMES, obs_values=obs_values)

    with trace() as tr, seed(rng_seed=jr.PRNGKey(0)):
        with Filter(filter_config=KFConfig(filter_source="cuthbert")):
            model()

    marginal_loglik = tr["f_marginal_loglik"]["value"]
    assert marginal_loglik.shape == (n_members,)
    assert jnp.isfinite(marginal_loglik).all()


# --- Gradients and parameter inference through time-varying callables ---


def test_gradient_flows_through_time_varying_cuthbert_kf():
    obs_values = _make_obs_values()

    def marginal_loglik(log_rate):
        def transition_matrix(t_now, t_next):
            return jsp_linalg.expm(-jnp.exp(log_rate) * jnp.eye(2) * (t_next - t_now))

        dynamics = _make_model(
            dict(A=transition_matrix, cov=_transition_cov), dict(H=H0, R=R0)
        )
        loglik, _ = compute_cuthbert_filter(
            dynamics,
            KFConfig(filter_source="cuthbert"),
            obs_times=IRREGULAR_TIMES,
            obs_values=obs_values,
        )
        return loglik

    gradient = jax.grad(marginal_loglik)(jnp.array(0.0))
    assert jnp.isfinite(gradient)


def _time_varying_nuts_model(
    obs_times=None, obs_values=None, ctrl_times=None, ctrl_values=None
):
    del ctrl_times, ctrl_values
    alpha = numpyro.sample("alpha", dist.Uniform(0.1, 1.0))

    def transition_matrix(t_now, t_next):
        drift_matrix = jnp.array([[-alpha, 0.4], [0.0, -0.3]])
        return jsp_linalg.expm(drift_matrix * (t_next - t_now))

    dynamics = DynamicalModel(
        initial_condition=dist.MultivariateNormal(jnp.zeros(2), jnp.eye(2)),
        state_evolution=LinearGaussianStateEvolution(
            A=transition_matrix, cov=_transition_cov
        ),
        observation_model=LinearGaussianObservation(H=H0, R=R0),
    )
    dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)


def test_nuts_smoke_time_varying_cuthbert_kf():
    obs_values = _make_obs_values()
    with Filter(filter_config=KFConfig(filter_source="cuthbert")):
        inference = MCMCInference(
            mcmc_config=NUTSConfig(
                num_samples=1, num_warmup=1, num_chains=1, mcmc_source="numpyro"
            ),
            model=_time_varying_nuts_model,
        )
        posterior_samples = inference.run(jr.PRNGKey(0), IRREGULAR_TIMES, obs_values)

    assert "alpha" in posterior_samples
    assert jnp.isfinite(posterior_samples["alpha"]).all()
