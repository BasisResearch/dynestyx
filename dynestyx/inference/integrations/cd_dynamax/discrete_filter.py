"""Discrete-time filters via cd-dynamax (dynamax): KF, EKF, UKF."""

import dataclasses
import warnings
from functools import partial
from typing import Unpack, cast

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm import inference as kf
from cd_dynamax.dynamax.nonlinear_gaussian_ssm import inference_ekf as ekf
from cd_dynamax.dynamax.nonlinear_gaussian_ssm import inference_ukf as ukf
from jax.experimental import sparse as jax_sparse
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.configs.filters import BaseFilterConfig, EKFConfig, KFConfig, UKFConfig
from dynestyx.inference.custom_filter import _legacy_filter_result, filter_history
from dynestyx.inference.filter_types import FilterObservations, SplitFilter
from dynestyx.inference.integrations.cd_dynamax.utils import (
    _require_constant_linear_gaussian_fields,
    gaussian_filter,
    gaussian_to_nlgssm_params,
)
from dynestyx.inference.integrations.utils import squeeze_leading_singletons
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)


def _lti_to_lgssm_params(dynamics: DynamicalModel):
    """Translate constant model matrices into the backend's typed parameter record."""
    evo, obs, prior = (
        dynamics.state_evolution,
        dynamics.observation_model,
        dynamics.initial_condition,
    )
    if not (
        isinstance(evo, LinearGaussianStateEvolution)
        and isinstance(obs, LinearGaussianObservation)
        and isinstance(prior, dist.MultivariateNormal)
    ):
        raise TypeError(
            "filter_type='kf' expects a DynamicalModel with LinearGaussianStateEvolution and LinearGaussianObservation and initial_condition as MultivariateNormal."
        )
    where = "The cd_dynamax discrete Kalman filter"
    _require_constant_linear_gaussian_fields(
        evo, ("A", "B", "bias", "cov"), where=where
    )
    _require_constant_linear_gaussian_fields(obs, ("H", "D", "bias", "R"), where=where)
    A, B, b, Q = evo.params_at(0, 1)
    H, D, d, R = obs.params_at(0)
    # Use the backend's zero defaults for absent control matrices and offsets.
    state_dim, obs_dim, control_dim = (
        dynamics.state_dim,
        dynamics.observation_dim,
        dynamics.control_dim,
    )
    return kf.ParamsLGSSM(
        initial=kf.ParamsLGSSMInitial(
            squeeze_leading_singletons(prior.loc, 1),
            squeeze_leading_singletons(prior.covariance_matrix, 2),
        ),
        dynamics=kf.ParamsLGSSMDynamics(
            weights=A,
            bias=jnp.zeros(state_dim) if b is None else b,
            input_weights=jnp.zeros((state_dim, control_dim)) if B is None else B,
            cov=Q,
        ),
        emissions=kf.ParamsLGSSMEmissions(
            weights=H,
            bias=jnp.zeros(obs_dim) if d is None else d,
            input_weights=jnp.zeros((obs_dim, control_dim)) if D is None else D,
            cov=R,
        ),
    )


def _prepare_inputs(
    dynamics: DynamicalModel,
    obs_values: Real[Array, "obs_time observation_dim"],
    obs_times: Real[Array, " obs_time"],
    ctrl_times: Real[Array, " ctrl_time"] | None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None,
) -> tuple[
    Real[Array, "obs_time observation_dim"],
    Real[Array, "obs_time control_dim"],
]:
    """Prepare emissions and inputs arrays for cd-dynamax discrete filters."""
    emissions = obs_values
    t1 = emissions.shape[0]
    control_dim = dynamics.control_dim
    if ctrl_values is None:
        inputs = jnp.zeros((t1, control_dim))
    elif ctrl_values.shape[0] > t1:
        aligned_ctrl_times = cast(Real[Array, " ctrl_time"], ctrl_times)
        inds = jnp.searchsorted(aligned_ctrl_times, obs_times, side="left")
        inputs = ctrl_values[inds]
    else:
        inputs = ctrl_values
    return emissions, inputs


def build_kf(dynamics: DynamicalModel) -> SplitFilter:
    params = _lti_to_lgssm_params(dynamics)
    F, b, B, Q = params.dynamics
    H, d, D, R = params.emissions

    def predict(moments, x):
        return kf._predict(*moments, F, B, b, Q, x.u_prev)

    def condition(moments, x):
        mean, cov = moments
        obs_mean = H @ mean + D @ x.u + d
        increment = kf.MVN(obs_mean, R + H @ cov @ cast(Array, H).T).log_prob(x.y)
        return increment, kf._condition_on(mean, cov, H, D, d, R, x.u, x.y), {}

    return gaussian_filter(
        (params.initial.mean, params.initial.cov),
        predict,
        condition,
    )


def build_ekf(dynamics: DynamicalModel) -> SplitFilter:
    params_nl = gaussian_to_nlgssm_params(dynamics)
    f, h = params_nl.dynamics_function, params_nl.emission_function
    Q, R = params_nl.dynamics_covariance, params_nl.emission_covariance
    if isinstance(dynamics.observation_model, LinearGaussianObservation) and isinstance(
        dynamics.observation_model.H, jax_sparse.JAXSparse
    ):
        warnings.warn(
            "A sparse observation matrix H was passed to EKFConfig. This works correctly, but likely gives no efficiency gain due to internal use of automatic differentiation.",
            stacklevel=2,
        )
    F, H = jax.jacfwd(f), jax.jacfwd(h)

    def predict(moments, x):
        return ekf._predict(*moments, f, F, Q, x.u_prev)

    def condition(moments, x):
        m, P = moments
        J = H(m, x.u)
        increment = ekf.MVN(h(m, x.u), J @ P @ J.T + R).log_prob(x.y)
        return increment, ekf._condition_on(m, P, h, H, R, x.u, x.y, 1), {}

    return gaussian_filter(
        (params_nl.initial_mean, params_nl.initial_covariance),
        predict,
        condition,
        save_forecasts=True,
    )


def build_ukf(dynamics: DynamicalModel, config: UKFConfig) -> SplitFilter:
    params_nl = gaussian_to_nlgssm_params(dynamics)
    f, h = params_nl.dynamics_function, params_nl.emission_function
    Q, R = params_nl.dynamics_covariance, params_nl.emission_covariance
    n = dynamics.state_dim
    scale = ukf._compute_lambda(config.alpha, config.kappa, n)
    wm, wc = ukf._compute_weights(n, config.alpha, config.beta, scale)

    def predict(moments, x):
        m, P, _ = ukf._predict(*moments, f, Q, scale, wm, wc, x.u_prev)
        return m, P

    def condition(moments, x):
        ll, m, P = ukf._condition_on(*moments, h, R, scale, wm, wc, x.u, x.y)
        return ll, (m, P), {}

    return gaussian_filter(
        (params_nl.initial_mean, params_nl.initial_covariance),
        predict,
        condition,
        save_forecasts=True,
    )


def compute_cd_dynamax_discrete_filter(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: PRNGKeyArray | None = None,
    **observations: Unpack[FilterObservations],
):
    if not isinstance(filter_config, (KFConfig, EKFConfig, UKFConfig)):
        raise ValueError(
            f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}"
        )
    if filter_config.filter_source != "cd_dynamax":
        overrides = (
            {"associative": False} if isinstance(filter_config, KFConfig) else {}
        )
        filter_config = dataclasses.replace(
            filter_config, filter_source="cd_dynamax", **overrides
        )
    return filter_history(dynamics, filter_config, key, **observations)[1].states


run_discrete_filter = partial(_legacy_filter_result, compute_cd_dynamax_discrete_filter)
