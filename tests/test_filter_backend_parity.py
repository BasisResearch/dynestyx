"""Compare native sequence, random-key, and forecast conventions for each backend."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from cd_dynamax import (
    ContDiscreteLinearGaussianSSM,
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_filter
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    unscented_kalman_filter,
)
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm.inference import (
    KFHyperParams,
)
from cuthbert.filtering import filter as cuthbert_filter

import dynestyx as dsx
from dynestyx import configs
from dynestyx.inference.custom_filter import step_inputs
from dynestyx.inference.filter_types import FilterInputs
from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (
    _lti_to_lgssm_params,
    compute_cd_dynamax_discrete_filter,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    dsx_to_cd_dynamax,
    dsx_to_cdlgssm_params,
    gaussian_to_nlgssm_params,
)
from dynestyx.inference.integrations.cuthbert.discrete_filter import (
    build_cuthbert_filter,
    run_cuthbert_steps,
)

TIMES = jnp.array([0.0, 0.1, 0.25, 0.3])
OBSERVATIONS = jnp.array([[0.1], [-0.5], [0.6], [0.3]])
CONTROLS = jnp.array([[0.2], [-0.3], [0.4], [-0.5]])
KEY = jax.random.PRNGKey(42)


def check_posterior(model, config, expected):
    with dsx.Filter(config):
        result = dsx.condition(
            "x",
            model,
            obs_times=TIMES,
            obs_values=OBSERVATIONS,
            ctrl_times=TIMES,
            ctrl_values=CONTROLS,
        )
    assert jax.tree.structure(result.states) == jax.tree.structure(expected)
    for (path, ref), got in zip(
        jax.tree.leaves_with_path(expected), jax.tree.leaves(result.states), strict=True
    ):
        np.testing.assert_allclose(got, ref, atol=2e-5, rtol=1e-4, err_msg=str(path))


@pytest.fixture
def model():
    return dsx.LTI_discrete(
        A=jnp.array([[0.9]]),
        Q=0.1 * jnp.eye(1),
        H=jnp.eye(1),
        R=0.2 * jnp.eye(1),
        B=jnp.eye(1),
    )


@pytest.mark.parametrize(
    "config",
    [
        configs.KFConfig(filter_source="cd_dynamax"),
        configs.EKFConfig(filter_source="cd_dynamax"),
        configs.UKFConfig(filter_source="cd_dynamax"),
    ],
)
def test_discrete_backend_forecasts(config, model):
    if isinstance(config, configs.KFConfig):
        expected = lgssm_filter(
            _lti_to_lgssm_params(model), OBSERVATIONS, inputs=CONTROLS
        )
    elif isinstance(config, configs.EKFConfig):
        expected = extended_kalman_filter(
            gaussian_to_nlgssm_params(model), OBSERVATIONS, inputs=CONTROLS
        )
    else:
        hp = UKFHyperParams(alpha=config.alpha, beta=config.beta, kappa=config.kappa)
        expected = unscented_kalman_filter(
            gaussian_to_nlgssm_params(model),
            OBSERVATIONS,
            hyperparams=hp,
            inputs=CONTROLS,
        )
    check_posterior(model, config, expected)


@pytest.mark.parametrize(
    "config",
    [
        configs.ContinuousTimeKFConfig(),
        configs.ContinuousTimeEKFConfig(),
        configs.ContinuousTimeUKFConfig(),
        configs.ContinuousTimeEnKFConfig(n_particles=8, crn_seed=KEY),
        configs.ContinuousTimeDPFConfig(n_particles=8, crn_seed=KEY),
    ],
)
def test_continuous_backend_moments_particles_and_observations(config):
    model = dsx.LTI_continuous(
        A=-jnp.eye(1),
        L=0.2 * jnp.eye(1),
        H=jnp.eye(1),
        R=0.2 * jnp.eye(1),
        B=jnp.eye(1),
    )
    common = dict(emissions=OBSERVATIONS, t_emissions=TIMES[:, None], inputs=CONTROLS)
    fields = [
        "marginal_loglik",
        "filtered_means",
        "filtered_covariances",
        "y_pred_mean",
        "y_pred_cov",
        "y_obs_pred_mean",
        "y_obs_pred_cov",
    ]
    solver = dict(dt0=config.diffeqsolve_dt0, max_steps=config.diffeqsolve_max_steps)
    if isinstance(config, configs.ContinuousTimeKFConfig):
        backend = ContDiscreteLinearGaussianSSM(1, 1, 1)
        expected = backend.filter(
            dsx_to_cdlgssm_params(model),
            **common,
            filter_hyperparams=KFHyperParams(diffeqsolve_settings=solver),
            output_fields=fields,
        )
    else:
        backend_cls = (
            ContDiscreteNonlinearSSM
            if isinstance(config, configs.ContinuousTimeDPFConfig)
            else ContDiscreteNonlinearGaussianSSM
        )
        backend = backend_cls(1, 1, 1)
        params, _ = dsx_to_cd_dynamax(model, cd_model=backend)
        options = dict(
            filter_state_order=config.filter_state_order,
            diffeqsolve_dt0=config.diffeqsolve_dt0,
            diffeqsolve_max_steps=config.diffeqsolve_max_steps,
            key=KEY,
        )
        if isinstance(config, configs.ContinuousTimeDPFConfig):
            options.update(
                filter_type="DPF",
                N_particles=config.n_particles,
                output_fields=None,
                extra_filter_kwargs=dict(
                    resample_method=config.resampling_method.differential_method,
                    softness=config.resampling_method.softness,
                    ess_threshold_ratio=config.ess_threshold_ratio,
                ),
            )
        elif isinstance(config, configs.ContinuousTimeEnKFConfig):
            options.update(
                filter_type="EnKF",
                enkf_N_particles=config.n_particles,
                enkf_inflation_delta=config.inflation_delta,
                output_fields=fields + ["y_ens_pred", "y_obs_ens_pred"],
            )
        elif isinstance(config, configs.ContinuousTimeUKFConfig):
            options.update(
                filter_type="UKF",
                output_fields=fields,
                extra_filter_kwargs=dict(
                    alpha=config.alpha, beta=config.beta, kappa=config.kappa
                ),
            )
        else:
            options.update(filter_type="EKF", filter_num_iter=1, output_fields=fields)
        expected = backend.filter(params, **common, **options)
    check_posterior(model, config, expected)


@pytest.mark.parametrize(
    "config",
    [
        configs.KFConfig(filter_source="cuthbert", associative=False),
        configs.KFConfig(filter_source="cuthbert", associative=True),
        configs.EKFConfig(filter_source="cuthbert"),
        configs.EnKFConfig(n_particles=12, crn_seed=KEY),
        configs.PFConfig(n_particles=12, crn_seed=KEY),
    ],
)
def test_cuthbert_native_sequence(config, model):
    inputs = FilterInputs(
        y=OBSERVATIONS,
        u=CONTROLS,
        u_prev=jnp.concatenate((CONTROLS[:1], CONTROLS[:-1])),
        time=TIMES,
        time_prev=jnp.concatenate((2 * TIMES[:1] - TIMES[1:2], TIMES[:-1])),
        is_first_step=jnp.arange(len(TIMES)) == 0,
    )
    algorithm = config.build_filter(model)
    init_key, scan_key = (
        (None, None) if config.crn_seed is None else jax.random.split(config.crn_seed)
    )
    raw = cuthbert_filter(
        algorithm,
        inputs,
        algorithm.init_prepare(key=init_key),
        parallel=isinstance(config, configs.KFConfig) and bool(config.associative),
        key=scan_key,
    )
    check_posterior(model, config, jax.tree.map(lambda xs: xs[1:], raw))


def test_explicit_cd_backend_accepts_default_kf_config(model):
    posterior = compute_cd_dynamax_discrete_filter(
        model,
        configs.KFConfig(),
        obs_times=TIMES,
        obs_values=OBSERVATIONS,
        ctrl_times=TIMES,
        ctrl_values=CONTROLS,
    )
    check_posterior(model, configs.KFConfig(filter_source="cd_dynamax"), posterior)


def test_online_particle_overrides(model):
    pf, parallel = build_cuthbert_filter(
        model,
        configs.PFConfig(),
        KEY,
        want_parallel=False,
        extra_filter_kwargs=dict(
            n_filter_particles=12,
            resampling_base_method="multinomial",
            ess_threshold=1.0,
        ),
    )
    inputs = step_inputs(model, TIMES, OBSERVATIONS, TIMES, CONTROLS)
    _, states = run_cuthbert_steps(pf, inputs, KEY, parallel=parallel)
    config = configs.PFConfig(
        crn_seed=KEY,
        n_particles=12,
        ess_threshold_ratio=1.0,
        resampling_method=configs.PFResamplingConfig(base_method="multinomial"),
    )
    check_posterior(model, config, states)
