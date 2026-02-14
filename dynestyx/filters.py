import dataclasses

import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM

from dynestyx.dynamical_models import Context, DynamicalModel
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
class FilterBasedMarginalLogLikelihood(BaseCDDynamaxLogFactorAdder):
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
        context: Context,
    ):
        """
        Add the marginal log likelihood as a numpyro factor.

        Args:
            name: Name of the factor.
            dynamics: Dynamical model to filter.
            context: Context containing the observations and controls.
        """
        config = (
            self.filter_config
            if self.filter_config is not None
            else _default_filter_config(dynamics)
        )

        filter_inputs = (name, dynamics, context, config)

        key = numpyro.prng_key()

        if dynamics.continuous_time:
            if not isinstance(config, ContinuousTimeConfigs):
                valid = [c.__name__ for c in ContinuousTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )
            _filter_continuous_time(*filter_inputs, key)
        else:
            if isinstance(config, HMMConfigs):
                _filter_hmm(*filter_inputs)  # type: ignore[arg-type]
            elif isinstance(config, DiscreteTimeConfigs):
                _filter_discrete_time(*filter_inputs)
            else:
                valid = [c.__name__ for c in HMMConfigs + DiscreteTimeConfigs]
                raise ValueError(
                    f"Invalid filter config: {type(config).__name__}. "
                    f"Valid config types: {valid}"
                )


__all__ = [
    "ContinuousTimeDPFConfig",
    "ContinuousTimeEnKFConfig",
    "ContinuousTimeEKFConfig",
    "ContinuousTimeUKFConfig",
    "EKFConfig",
    "EnKFConfig",
    "FilterBasedMarginalLogLikelihood",
    "HMMConfig",
    "HMMConfigs",
    "KFConfig",
    "PFConfig",
    "PFResamplingConfig",
    "UKFConfig",
]
