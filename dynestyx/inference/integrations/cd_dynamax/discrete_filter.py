"""Discrete-time filters via cd-dynamax (dynamax): KF, EKF, UKF."""

import warnings
from typing import Any, cast

import jax.numpy as jnp
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import (
    lgssm_filter,
)
from cd_dynamax.dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_filter,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    unscented_kalman_filter,
)
from jax.experimental import sparse as jax_sparse
from jaxtyping import Array, Real

from dynestyx.inference.configs.filter import (
    BaseFilterConfig,
    EKFConfig,
    KFConfig,
    UKFConfig,
)
from dynestyx.inference.integrations.cd_dynamax.utils import (
    _require_constant_linear_gaussian_fields,
    gaussian_to_nlgssm_params,
)
from dynestyx.inference.integrations.utils import squeeze_leading_singletons
from dynestyx.inference.utils.distribution_utils import _posterior_sequence_to_dists
from dynestyx.models import (
    DynamicalModel,
    LinearGaussianObservation,
    LinearGaussianStateEvolution,
)


def _lti_to_lgssm_params(dynamics: DynamicalModel):
    """Build dynamax ParamsLGSSM from LinearGaussianSSM.initialize for an LTI model."""
    state_dim = dynamics.state_dim
    emission_dim = dynamics.observation_dim
    control_dim = dynamics.control_dim

    if (
        isinstance(dynamics.state_evolution, LinearGaussianStateEvolution)
        and isinstance(dynamics.observation_model, LinearGaussianObservation)
        and isinstance(dynamics.initial_condition, dist.MultivariateNormal)
    ):
        evo = dynamics.state_evolution
        obs = dynamics.observation_model
        ic = dynamics.initial_condition
        _require_constant_linear_gaussian_fields(
            evo,
            ("A", "B", "bias", "cov"),
            where="The cd_dynamax discrete Kalman filter",
        )
        _require_constant_linear_gaussian_fields(
            obs,
            ("H", "D", "bias", "R"),
            where="The cd_dynamax discrete Kalman filter",
        )
        model = LinearGaussianSSM(
            state_dim=state_dim,
            emission_dim=emission_dim,
            input_dim=control_dim,
            has_dynamics_bias=evo.bias is not None,
            has_emissions_bias=obs.bias is not None,
        )
        params, _ = model.initialize(
            initial_mean=squeeze_leading_singletons(ic.loc, 1),
            initial_covariance=squeeze_leading_singletons(ic.covariance_matrix, 2),
            dynamics_weights=evo.A,
            dynamics_bias=evo.bias,
            dynamics_input_weights=evo.B,
            dynamics_covariance=evo.cov,
            emission_weights=obs.H,
            emission_bias=obs.bias,
            emission_input_weights=obs.D,
            emission_covariance=obs.R,
        )
        return params
    raise TypeError(
        "filter_type='kf' expects a DynamicalModel with LinearGaussianStateEvolution and LinearGaussianObservation and initial_condition as MultivariateNormal."
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


def compute_cd_dynamax_discrete_filter(
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
) -> Any:
    """Pure-JAX cd-dynamax discrete filter computation (no numpyro side-effects)."""
    emissions, inputs = _prepare_inputs(
        dynamics, obs_values, obs_times, ctrl_times, ctrl_values
    )

    if isinstance(filter_config, KFConfig):
        params = _lti_to_lgssm_params(dynamics)
        return lgssm_filter(params, emissions, inputs=inputs)

    # EKF and UKF share the same nonlinear params representation.
    params_nl = gaussian_to_nlgssm_params(dynamics)

    if isinstance(filter_config, EKFConfig):
        obs_model = dynamics.observation_model
        if isinstance(obs_model, LinearGaussianObservation) and isinstance(
            obs_model.H, jax_sparse.JAXSparse
        ):
            warnings.warn(
                "A sparse observation matrix H was passed to EKFConfig. This works "
                  "correctly, but likely gives no efficiency gain due to internal"
                  "use of automatic differentiation.",
                stacklevel=2,
            )
        return extended_kalman_filter(params_nl, emissions, inputs=inputs)
    if isinstance(filter_config, UKFConfig):
        hyperparams = UKFHyperParams(
            alpha=filter_config.alpha,
            beta=filter_config.beta,
            kappa=filter_config.kappa,
        )
        return unscented_kalman_filter(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )
    raise ValueError(
        f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
        "Expected KFConfig, EKFConfig, or UKFConfig."
    )


def run_discrete_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
    **kwargs,
) -> tuple[Real[Array, ""], object, list[dist.Distribution]]:
    """Run discrete-time filter via cd-dynamax (KF, EKF, UKF).

    Pure computation — no numpyro side-effects. Callers are responsible for
    registering numpyro.factor / numpyro.deterministic if needed.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}).
            - posterior: CD-Dynamax posterior object with filtered_means and
              filtered_covariances attributes.
            - filtered_dists: list of MultivariateNormal distributions p(x_t | y_{1:t})
              at each obs time, for posterior rollout.
    """
    posterior = compute_cd_dynamax_discrete_filter(
        dynamics,
        filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    filtered_dists = _posterior_sequence_to_dists(
        posterior,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=False,
        missing="empty",
    )
    return posterior.marginal_loglik, posterior, filtered_dists


__all__ = [
    "compute_cd_dynamax_discrete_filter",
    "run_discrete_filter",
    "_lti_to_lgssm_params",
    "_prepare_inputs",
]
