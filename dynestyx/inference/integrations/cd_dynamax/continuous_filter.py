"""Continuous-time filters via CD-Dynamax: KF, EnKF, DPF, EKF, UKF."""

from typing import Any

import numpyro.distributions as dist
from cd_dynamax import (
    ContDiscreteNonlinearGaussianSSM,
    ContDiscreteNonlinearSSM,
)
from jaxtyping import Array, PRNGKeyArray, Real

from dynestyx.configs.filters import (
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
)
from dynestyx.inference.custom_filter import step_inputs
from dynestyx.inference.utils.distribution_utils import _posterior_sequence_to_dists
from dynestyx.models import DynamicalModel

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM

ContinuousTimeFilterConfig = (
    ContinuousTimeKFConfig
    | ContinuousTimeEnKFConfig
    | ContinuousTimeDPFConfig
    | ContinuousTimeEKFConfig
    | ContinuousTimeUKFConfig
)


def compute_continuous_filter(
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
) -> Any:
    """Pure-JAX continuous-time filter computation (no numpyro side-effects)."""
    inputs = step_inputs(dynamics, obs_times, obs_values, ctrl_times, ctrl_values)
    if key is None and isinstance(
        filter_config, (ContinuousTimeEnKFConfig, ContinuousTimeDPFConfig)
    ):
        raise ValueError(
            f"{type(filter_config).__name__} requires a PRNG key: set 'crn_seed' in the filter config, or run inside a NumPyro seeded context."
        )
    return filter_config.build_filter(dynamics).run(inputs, key)


def run_continuous_filter(
    name: str,
    dynamics: DynamicalModel,
    filter_config: ContinuousTimeFilterConfig,
    key: PRNGKeyArray | None = None,
    *,
    obs_times: Real[Array, " obs_time"],
    obs_values: Real[Array, "obs_time observation_dim"],
    ctrl_times: Real[Array, " ctrl_time"] | None = None,
    ctrl_values: Real[Array, "ctrl_time control_dim"] | None = None,
    **kwargs,
) -> tuple[Real[Array, ""], object, list[dist.Distribution]]:
    """Run continuous-time filter via CD-Dynamax.

    Pure computation — no numpyro side-effects. Callers are responsible for
    registering numpyro.factor / numpyro.deterministic if needed.

    Returns:
        tuple of:
            - marginal_loglik: scalar marginal log-likelihood log p(y_{1:T}).
            - filtered_posterior: CD-Dynamax posterior object with filtered_means,
              filtered_covariances, and marginal_loglik attributes.
            - filtered_dists: list of MultivariateNormal distributions p(x_t | y_{1:t})
              at each obs time, for posterior rollout.
    """
    filtered = compute_continuous_filter(
        dynamics,
        filter_config,
        key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
    )

    filtered_dists = _posterior_sequence_to_dists(
        filtered,
        means_attr="filtered_means",
        covariances_attr="filtered_covariances",
        particle_mode=isinstance(filter_config, ContinuousTimeDPFConfig),
        missing_message="Filtered means/covariances unexpectedly None for non-DPF config",
    )
    return filtered.marginal_loglik, filtered, filtered_dists


__all__ = [
    "ContinuousTimeFilterConfig",
    "compute_continuous_filter",
    "run_continuous_filter",
]
