"""Continuous-discrete filtering steps using CD-Dynamax numerical kernels."""

import jax
import jax.numpy as jnp
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
from cd_dynamax.src.continuous_discrete_nonlinear_ssm import inference_dpf as dpf
from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import (
    PosteriorCDNLSSMFiltered,
)

from dynestyx.inference.integrations.cd_dynamax.filter_steps import (
    CDFilter,
    gaussian_filter,
)


def _diagnostics(enabled, mean, cov, obs_mean, obs_cov):
    if enabled:
        return dict(
            y_pred_mean=mean,
            y_pred_cov=cov,
            y_obs_pred_mean=obs_mean,
            y_obs_pred_cov=obs_cov,
        )
    return {}


def linear_filter(params, hp, diagnostics, warn, control_dim):
    params, _ = kf.preprocess_params_and_inputs(params, 0, jnp.empty((0, control_dim)))

    def predict(moments, x):
        t = x.time_prev
        B = kf._get_params(params.dynamics.input_weights, 2, t)
        b = kf._get_params(params.dynamics.bias, 1, t)
        F, Q, C = kf.compute_pushforward(
            params, t, x.time, diffeqsolve_settings=hp.diffeqsolve_settings, warn=warn
        )
        return kf._predict(*moments, F, C, B, b, Q, x.u_prev, warn=warn)

    def condition(moments, x):
        t = x.time
        H = kf._get_params(params.emissions.weights, 2, t)
        D = kf._get_params(params.emissions.input_weights, 2, t)
        d = kf._get_params(params.emissions.bias, 1, t)
        R = kf._get_params(params.emissions.cov, 2, t)
        mean, cov = kf._emission_predicted_moments(*moments, H, D, d, x.u, warn=warn)
        obs_cov = kf.psd(cov + R, warn=warn)
        ll = kf.MVN(mean, obs_cov).log_prob(x.y)
        filtered = kf._condition_on(*moments, H, D, d, R, x.u, x.y, warn=warn)
        return ll, filtered, _diagnostics(diagnostics, mean, cov, mean, obs_cov)

    return gaussian_filter(
        (params.initial.mean, params.initial.cov),
        predict,
        condition,
        kf.PosteriorGSSMFiltered,
    )


def nonlinear_filter(params, hp, diagnostics, warn):
    initial = (params.initial.mean.f(), params.initial.cov.f())
    h = params.emissions.emission_function.f
    if isinstance(hp, ekf.EKFHyperParams):
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
    else:
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

    return gaussian_filter(initial, predict, condition, ekf.PosteriorGSSMFiltered)


def ensemble_filter(params, hp, diagnostics, warn):
    def initialize(x, *, key):
        ensemble = jax.random.multivariate_normal(
            key,
            params.initial.mean.f(),
            params.initial.cov.f(),
            shape=(hp.N_particles,),
        )
        return ensemble, jnp.zeros((), dtype=ensemble.dtype)

    def predict(filtered, x, *, key):
        ensemble, ll, _ = filtered
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
        fields = (
            {
                name: analysis[name]
                for name in (
                    "y_pred_mean",
                    "y_pred_cov",
                    "y_obs_pred_mean",
                    "y_obs_pred_cov",
                    "y_ens_pred",
                    "y_obs_ens_pred",
                )
            }
            if diagnostics
            else {}
        )
        return analysis["x_cond"], ll + analysis["loglik_step"], fields

    def gather(prior, filtered):
        ensemble, ll, fields = filtered
        mean = jnp.mean(ensemble, axis=0)
        centered = ensemble - mean
        cov = enkf.psd(
            jnp.sum(enkf._outer(centered, centered), axis=0) / (hp.N_particles - 1),
            warn=warn,
        )
        return ll, dict(filtered_means=mean, filtered_covariances=cov, **fields)

    return CDFilter(
        initialize,
        predict,
        update,
        gather,
        enkf.PosteriorGSSMFiltered,
        ensemble_keys=True,
    )


def particle_filter(params, hp):
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
        return (next_particles, next_weights, ll + increment), (particles, weights)

    def gather(prior, filtered):
        carry, (particles, log_w) = filtered
        weights = jnp.exp(log_w)
        mean, cov = dpf.dpf_moments(particles, weights / jnp.sum(weights))
        return carry[2], dict(
            filtered_means=mean,
            filtered_covariances=cov,
            particles=particles,
            log_weights=log_w,
        )

    return CDFilter(initialize, predict, update, gather, PosteriorCDNLSSMFiltered)
