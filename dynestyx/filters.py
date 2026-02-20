import dataclasses

import jax
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM

from dynestyx.dynamical_models import DynamicalModel
from dynestyx.handlers import BaseCDDynamaxLogFactorAdder
from dynestyx.inference.continuous_time_filters import (
    _filter_continuous_time,
)
from dynestyx.inference.discrete_time_filters import _filter_discrete_time
from dynestyx.inference.filter_configs import (
    BaseFilterConfig,
    ContinuousTimeConfigs,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeEnKFConfig,
    ContinuousTimeKFConfig,
    ContinuousTimeUKFConfig,
    DiscreteTimeConfigs,
    EKFConfig,
    EnKFConfig,
    HMMConfig,
    HMMConfigs,
    KFConfig,
    PFConfig,
    PFResamplingConfig,
    UKFConfig,
)
from dynestyx.inference.hmm_filters import _filter_hmm

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


def _default_filter_config(dynamics: DynamicalModel):
    """Return appropriate default filter config when none specified."""
    if dynamics.continuous_time:
        return ContinuousTimeEnKFConfig()

    # default to particle filter in discrete time
    return EKFConfig(filter_source="cuthbert")


@dataclasses.dataclass
class Filter(BaseCDDynamaxLogFactorAdder):
    """
    Object for filtering a dynamical model, and adding the resulting marginal log likelihood as a numpyro factor.

    Uses a single filter_config to specify the filter. If None, defaults are chosen:
    - Continuous-time: Ensemble Kalman Filter (ContinuousTimeEnKFConfig)
    - Discrete-time: Extended Kalman Filter (EKFConfig)

    Args:
        filter_config: Filter configuration. If None, defaults are chosen.

    For HMMs, must use `HMMConfig` to specify the filter.

    """

    filter_config: BaseFilterConfig | None = None

    def add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times: jax.Array | None = None,
        obs_values: jax.Array | None = None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ):
        """
        Add the marginal log likelihood as a numpyro factor.

        Args:
            name: Name of the factor.
            dynamics: Dynamical model to filter.
            obs_times: Observation times.
            obs_values: Observed values.
            ctrl_times: Control times (optional).
            ctrl_values: Control values (optional).
        """
        if obs_times is None or obs_values is None:
            raise ValueError("obs_times and obs_values are required for filtering.")

        config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )

        key = numpyro.prng_key() if config.crn_seed is None else config.crn_seed

        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            _filter_continuous_time(
                name,
                dynamics,
                config,  # type: ignore[arg-type]
                key=key,
                obs_times=obs_times,
                obs_values=obs_values,
                ctrl_times=ctrl_times,
                ctrl_values=ctrl_values,
                **kwargs,
            )
        else:
            if isinstance(config, HMMConfigs):
                _filter_hmm(
                    name,
                    dynamics,
                    config,  # type: ignore[arg-type]
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    **kwargs,
                )
            elif isinstance(config, DiscreteTimeConfigs):
                _filter_discrete_time(
                    name,
                    dynamics,
                    config,  # type: ignore[arg-type]
                    key=key,
                    obs_times=obs_times,
                    obs_values=obs_values,
                    ctrl_times=ctrl_times,
                    ctrl_values=ctrl_values,
                    **kwargs,
                )
            else:
                valid = [c.__name__ for c in HMMConfigs + DiscreteTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )


__all__ = [
    "ContinuousTimeKFConfig",
    "ContinuousTimeDPFConfig",
    "ContinuousTimeEnKFConfig",
    "ContinuousTimeEKFConfig",
    "ContinuousTimeUKFConfig",
    "EKFConfig",
    "EnKFConfig",
    "Filter",
    "HMMConfig",
    "HMMConfigs",
    "KFConfig",
    "PFConfig",
    "PFResamplingConfig",
    "UKFConfig",
]
