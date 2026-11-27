"""Model-bound continuous-discrete filters using CD-Dynamax numerical kernels."""

from functools import partial
from typing import Unpack

import jax
import jax.numpy as jnp
from cd_dynamax import ContDiscreteNonlinearSSM
from cd_dynamax.src.continuous_discrete_linear_gaussian_ssm import inference as kf
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    inference_ekf as ekf,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    inference_enkf as enkf,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm import (
    inference_ukf as ukf,
)
from cd_dynamax.src.continuous_discrete_nonlinear_gaussian_ssm.models import (
    build_filter_hyperparams,
)
from cd_dynamax.src.continuous_discrete_nonlinear_ssm import inference_dpf as dpf
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.inference_dpf import (
    build_dpf_hyperparams,
)
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import (
    PosteriorCDNLSSMFiltered,
)
from jaxtyping import PRNGKeyArray

from dynestyx.configs.filters import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
)
from dynestyx.inference.custom_filter import _legacy_filter_result, filter_history
from dynestyx.inference.filter_types import FilterObservations, SplitFilter
from dynestyx.inference.integrations.cd_dynamax.utils import (
    dsx_to_cd_dynamax,
    dsx_to_cdlgssm_params,
    gaussian_filter,
)
from dynestyx.inference.observation_predictions import (
    wants_observation_prediction_diagnostics,
)
from dynestyx.models import DynamicalModel

ContinuousTimeFilterConfig = (
    ContinuousTimeKFConfig
    | ContinuousTimeEnKFConfig
    | ContinuousTimeDPFConfig
    | ContinuousTimeEKFConfig
    | ContinuousTimeUKFConfig
)


def _solver_kwargs(config: ContinuousTimeFilterConfig):
    return {
        "filter_state_order": config.filter_state_order,
        "filter_state_cov_rescaling": config.cov_rescaling
        if config.cov_rescaling is not None
        else 1.0,
        "diffeqsolve_max_steps": config.diffeqsolve_max_steps,
        "diffeqsolve_dt0": config.diffeqsolve_dt0,
        "diffeqsolve_kwargs": config.diffeqsolve_kwargs,
    }


def _diagnostics(enabled, mean, cov, obs_mean, obs_cov):
    if enabled:
        return dict(
            y_pred_mean=mean,
            y_pred_cov=cov,
            y_obs_pred_mean=obs_mean,
            y_obs_pred_cov=obs_cov,
        )
    return {}


def build_kf(model: DynamicalModel, config: ContinuousTimeKFConfig) -> SplitFilter:
    params = dsx_to_cdlgssm_params(model)
    solver = {
        "dt0": config.diffeqsolve_dt0,
        "max_steps": config.diffeqsolve_max_steps,
        **config.diffeqsolve_kwargs,
    }
    warn = config.warn
    diagnostics = wants_observation_prediction_diagnostics(config)
    # dsx_to_cdlgssm_params resolves the constant matrices, including zero
    # control weights and biases when the model omits them.
    B, b = params.dynamics.input_weights, params.dynamics.bias
    H, D = params.emissions.weights, params.emissions.input_weights
    d, R = params.emissions.bias, params.emissions.cov

    def predict(moments, x):
        F, Q, C = kf.compute_pushforward(
            params, x.time_prev, x.time, diffeqsolve_settings=solver, warn=warn
        )
        return kf._predict(*moments, F, C, B, b, Q, x.u_prev, warn=warn)

    def condition(moments, x):
        mean, cov = kf._emission_predicted_moments(*moments, H, D, d, x.u, warn=warn)
        obs_cov = kf.psd(cov + R, warn=warn)
        ll = kf.MVN(mean, obs_cov).log_prob(x.y)
        filtered = kf._condition_on(*moments, H, D, d, R, x.u, x.y, warn=warn)
        return ll, filtered, _diagnostics(diagnostics, mean, cov, mean, obs_cov)

    return gaussian_filter(
        (params.initial.mean, params.initial.cov),
        predict,
        condition,
    )


def build_ekf(model: DynamicalModel, config: ContinuousTimeEKFConfig) -> SplitFilter:
    params, _ = dsx_to_cd_dynamax(model)
    hp = build_filter_hyperparams(
        filter_type="EKF",
        filter_emission_order=config.filter_emission_order,
        extra_filter_kwargs=config.extra_filter_kwargs,
        **_solver_kwargs(config),
    )
    warn = config.warn
    diagnostics = wants_observation_prediction_diagnostics(config)
    initial = (params.initial.mean.f(), params.initial.cov.f())
    h = params.emissions.emission_function.f
    H = jax.jacfwd(h)

    def predict(moments, x):
        return ekf._predict(
            *moments, params, x.time_prev, x.time, x.u_prev, hp, warn=warn
        )

    def condition(moments, x):
        R = params.emissions.emission_cov.f(None, x.u, x.time)
        mean, cov = ekf._emission_predicted_moments(
            *moments, h, H, x.u, x.time, warn=warn
        )
        obs_cov = ekf.psd(cov + R, warn=warn)
        ll = ekf.MVN(mean, obs_cov).log_prob(x.y)
        filtered = ekf._condition_on(
            *moments, h, H, R, x.u, x.y, x.time, 1, hp, warn=warn
        )
        return ll, filtered, _diagnostics(diagnostics, mean, cov, mean, obs_cov)

    return gaussian_filter(initial, predict, condition)


def build_ukf(model: DynamicalModel, config: ContinuousTimeUKFConfig) -> SplitFilter:
    params, _ = dsx_to_cd_dynamax(model)
    hp = build_filter_hyperparams(
        filter_type="UKF",
        extra_filter_kwargs={
            "alpha": config.alpha,
            "beta": config.beta,
            "kappa": config.kappa,
            **config.extra_filter_kwargs,
        },
        **_solver_kwargs(config),
    )
    warn = config.warn
    diagnostics = wants_observation_prediction_diagnostics(config)
    initial = (params.initial.mean.f(), params.initial.cov.f())
    h = params.emissions.emission_function.f
    n = params.dynamics.diffusion_cov.f(None, None, None).shape[0]
    scale = ukf._compute_lambda(hp.alpha, hp.kappa, n)
    wm, wc, W = ukf._compute_weights(n, hp.alpha, hp.beta, scale)

    def predict(moments, x):
        return ukf._predict(
            *moments,
            params,
            x.time_prev,
            x.time,
            scale,
            wm,
            wc,
            W,
            x.u_prev,
            hp,
            warn=warn,
        )

    def condition(moments, x):
        R = params.emissions.emission_cov.f(None, x.u, x.time)
        ll, m, P, mean, cov, obs_mean, obs_cov = ukf._condition_on(
            *moments, h, R, scale, wm, wc, x.u, x.y, x.time, warn=warn
        )
        return ll, (m, P), _diagnostics(diagnostics, mean, cov, obs_mean, obs_cov)

    return gaussian_filter(initial, predict, condition)


def build_enkf(model: DynamicalModel, config: ContinuousTimeEnKFConfig) -> SplitFilter:
    params, _ = dsx_to_cd_dynamax(model)
    hp = build_filter_hyperparams(
        filter_type="EnKF",
        enkf_N_particles=config.n_particles,
        enkf_inflation_delta=config.inflation_delta,
        extra_filter_kwargs={
            "perturb_measurements": True
            if config.perturb_measurements is None
            else config.perturb_measurements,
            **config.extra_filter_kwargs,
        },
        **_solver_kwargs(config),
    )
    warn = config.warn
    diagnostics = wants_observation_prediction_diagnostics(config)

    observation_fields = (
        (
            "y_pred_mean",
            "y_pred_cov",
            "y_obs_pred_mean",
            "y_obs_pred_cov",
            "y_ens_pred",
            "y_obs_ens_pred",
        )
        if diagnostics
        else ()
    )

    def initialize(x, *, key):
        ensemble = jax.random.multivariate_normal(
            key,
            params.initial.mean.f(),
            params.initial.cov.f(),
            shape=(hp.N_particles,),
        )
        return ensemble, jnp.zeros((), dtype=ensemble.dtype)

    def predict(filtered, x, *, key):
        ensemble, posterior = filtered
        ll = posterior.marginal_loglik
        return enkf._predict(
            key, ensemble, params, x.time_prev, x.time, x.u_prev, hp
        ), ll

    def update(prior, x, *, key):
        ensemble, ll = prior
        R = params.emissions.emission_cov.f(None, x.u, x.time)
        analysis = enkf._condition_on(
            key,
            ensemble,
            params.emissions.emission_function.f,
            R,
            x.u,
            x.y,
            x.time,
            hp.perturb_measurements,
            hp.inflation_delta,
            warn=warn,
        )
        fields = {name: analysis[name] for name in observation_fields}
        ensemble = analysis["x_cond"]
        ll = ll + analysis["loglik_step"]
        mean = jnp.mean(ensemble, axis=0)
        centered = ensemble - mean
        cov = enkf.psd(
            jnp.sum(enkf._outer(centered, centered), axis=0) / (hp.N_particles - 1),
            warn=warn,
        )
        return ensemble, enkf.PosteriorGSSMFiltered(
            marginal_loglik=ll, filtered_means=mean, filtered_covariances=cov, **fields
        )

    return SplitFilter(initialize, predict, update)


def build_dpf(model: DynamicalModel, config: ContinuousTimeDPFConfig) -> SplitFilter:
    if config.resampling_method.base_method != "multinomial":
        raise ValueError("Only multinomial resampling is supported for CD-Dynamax DPF.")
    backend = ContDiscreteNonlinearSSM(
        state_dim=model.state_dim,
        emission_dim=model.observation_dim,
        input_dim=model.control_dim,
    )
    params, _ = dsx_to_cd_dynamax(model, cd_model=backend)
    hp = build_dpf_hyperparams(
        N_particles=config.n_particles,
        extra_filter_kwargs={
            "resample_method": config.resampling_method.differential_method,
            "softness": config.resampling_method.softness,
            "ess_threshold_ratio": config.ess_threshold_ratio,
            **config.extra_filter_kwargs,
        },
        **_solver_kwargs(config),
    )
    method = dpf._validate_resample_method(hp.resample_method)
    if hp.proposal_method != "bootstrap":
        raise ValueError("Only bootstrap proposals are supported.")

    def initialize(x, *, key):
        particles = params.initial.initial_distribution.sample(
            seed=key, sample_shape=(hp.N_particles,)
        )
        return (
            particles,
            jnp.full((hp.N_particles,), -jnp.log(hp.N_particles)),
            jnp.array(0.0),
        )

    def predict(filtered, x, *, key):
        particles, weights, ll = filtered[0]
        particles = dpf._predict(
            key,
            particles,
            params,
            t0=x.time_prev,
            t1=x.time,
            u=x.u_prev,
            filter_hyperparams=hp,
        )
        return particles, weights, ll

    def update(prior, x, *, key):
        particles, log_w, ll = prior
        likelihood = params.emissions.emission_distribution.log_prob(
            x.y, particles, x.u, x.time
        )
        weights, increment = dpf._normalize_log_weights(log_w + likelihood)

        def resample():
            if method == "soft":
                return dpf._soft_resample(key, particles, weights, hp.softness)
            if method == "multinomial":
                return dpf._multinomial_resample(key, particles, weights)[:2]
            return dpf._stop_gradient_resample(
                key, particles, weights, base_method="multinomial"
            )

        next_particles, next_weights = jax.lax.cond(
            dpf._effective_sample_size(weights)
            < hp.ess_threshold_ratio * hp.N_particles,
            resample,
            lambda: (particles, weights),
        )
        probabilities = jnp.exp(weights)
        mean, cov = dpf.dpf_moments(particles, probabilities / jnp.sum(probabilities))
        posterior = PosteriorCDNLSSMFiltered(
            filtered_means=mean,
            filtered_covariances=cov,
            particles=particles,
            log_weights=weights,
            marginal_loglik=ll + increment,
        )
        return (next_particles, next_weights, ll + increment), posterior

    return SplitFilter(initialize, predict, update)


def compute_continuous_filter(
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: PRNGKeyArray | None = None,
    **observations: Unpack[FilterObservations],
):
    return filter_history(dynamics, filter_config, key, **observations)[1]


run_continuous_filter = partial(_legacy_filter_result, compute_continuous_filter)
