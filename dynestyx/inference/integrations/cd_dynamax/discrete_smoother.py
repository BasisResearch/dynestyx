"""Discrete-time smoothers via cd-dynamax (dynamax): KF, EKF, UKF."""

import jax
import numpyro.distributions as dist
from cd_dynamax.dynamax.linear_gaussian_ssm.inference import lgssm_smoother
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ekf import (
    extended_kalman_smoother,
)
from cd_dynamax.dynamax.nonlinear_gaussian_ssm.inference_ukf import (
    UKFHyperParams,
    unscented_kalman_smoother,
)

from dynestyx.inference.distribution_utils import _posterior_sequence_to_dists
from dynestyx.inference.filter_configs import (
    EKFConfig,
    KFConfig,
    UKFConfig,
)
from dynestyx.inference.integrations.cd_dynamax.discrete_filter import (
    _lti_to_lgssm_params,
    _prepare_inputs,
)
from dynestyx.inference.integrations.cd_dynamax.utils import gaussian_to_nlgssm_params
from dynestyx.inference.smoother_configs import (
    BaseSmootherConfig,
)
from dynestyx.models import DynamicalModel


def compute_cd_dynamax_discrete_smoother(
    dynamics: DynamicalModel,
    filter_config: BaseSmootherConfig,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
):
    """Pure-JAX cd-dynamax discrete smoother computation (no numpyro side-effects)."""
    emissions, inputs = _prepare_inputs(
        dynamics, obs_values, obs_times, ctrl_times, ctrl_values
    )

    if isinstance(filter_config, KFConfig):
        params = _lti_to_lgssm_params(dynamics)
        return lgssm_smoother(params, emissions, inputs=inputs)

    params_nl = gaussian_to_nlgssm_params(dynamics)
    if isinstance(filter_config, EKFConfig):
        return extended_kalman_smoother(params_nl, emissions, inputs=inputs)
    if isinstance(filter_config, UKFConfig):
        hyperparams = UKFHyperParams(
            alpha=filter_config.alpha,
            beta=filter_config.beta,
            kappa=filter_config.kappa,
        )
        return unscented_kalman_smoother(
            params_nl, emissions, hyperparams=hyperparams, inputs=inputs
        )
    raise ValueError(
        f"Unsupported cd-dynamax discrete config: {type(filter_config).__name__}. "
        "Expected KFConfig, EKFConfig, or UKFConfig."
    )


def run_discrete_smoother(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseSmootherConfig,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> tuple[jax.Array, object, list[dist.Distribution]]:
    """Run discrete-time smoother via cd-dynamax (KF, EKF, UKF).

    Pure computation — no numpyro side-effects. Callers are responsible for
    registering numpyro.factor / numpyro.deterministic if needed.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}).
            - posterior: CD-Dynamax posterior object with smoothed_means and
              smoothed_covariances attributes.
            - smoothed_dists: list of MultivariateNormal distributions p(x_t | y_{1:T})
              at each obs time, for posterior rollout.
    """
    posterior = compute_cd_dynamax_discrete_smoother(
        dynamics,
        filter_config,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    smoothed_dists = _posterior_sequence_to_dists(
        posterior,
        means_attr="smoothed_means",
        covariances_attr="smoothed_covariances",
        particle_mode=False,
        missing="empty",
    )
    return posterior.marginal_loglik, posterior, smoothed_dists


__all__ = ["compute_cd_dynamax_discrete_smoother", "run_discrete_smoother"]
