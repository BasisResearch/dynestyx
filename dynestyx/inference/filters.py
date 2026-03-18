import dataclasses

import jax
import numpyro
from cd_dynamax import ContDiscreteNonlinearGaussianSSM, ContDiscreteNonlinearSSM
from effectful.ops.semantics import fwd
from effectful.ops.syntax import ObjectInterpretation, implements

from dynestyx.handlers import HandlesSelf, _sample_intp
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
    MarginalPFConfig,
    PFConfig,
    PFResamplingConfig,
    UKFConfig,
)
from dynestyx.inference.hmm_filters import _filter_hmm
from dynestyx.inference.integrations.cd_dynamax.continuous import run_continuous_filter
from dynestyx.inference.integrations.cd_dynamax.discrete import (
    run_discrete_filter as run_cd_dynamax_discrete,
)
from dynestyx.inference.integrations.cuthbert.discrete import (
    run_discrete_filter as run_cuthbert_discrete,
)
from dynestyx.inference.integrations.pfjax.discrete import (
    run_discrete_filter as run_pfjax_discrete,
)
from dynestyx.models import DynamicalModel
from dynestyx.types import FunctionOfTime

type SSMType = ContDiscreteNonlinearGaussianSSM | ContDiscreteNonlinearSSM


class BaseLogFactorAdder(ObjectInterpretation, HandlesSelf):
    """Base for filter handlers."""

    @implements(_sample_intp)
    def _sample_ds(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ) -> FunctionOfTime:
        self._add_log_factors(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
        # Forward unchanged so downstream handlers (or default implementation)
        # can still see this op if needed.
        return fwd(
            name,
            dynamics,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )

    def _add_log_factors(
        self,
        name: str,
        dynamics: DynamicalModel,
        *,
        obs_times=None,
        obs_values=None,
        ctrl_times=None,
        ctrl_values=None,
        **kwargs,
    ):
        # Inheritors should implement this method.
        raise NotImplementedError()


def _default_filter_config(dynamics: DynamicalModel):
    """Return appropriate default filter config when none specified."""
    if dynamics.continuous_time:
        return ContinuousTimeEnKFConfig()

    # default to particle filter in discrete time
    return EKFConfig(filter_source="cuthbert")


@dataclasses.dataclass
class Filter(BaseLogFactorAdder):
    r"""Performs Bayesian filtering to compute the filtering distribution $p(x_t | y_{1:t})$ and the marginal likelihood $\log p(y_{1:T})$.

    A `Filter` object should be used as a context manager around a call to a model with a `dsx.sample(...)` statement
    to condition a dynamical model on observations via a filtering algorithm. The filter
    is selected and dispatched according to the `filter_config` argument, which adds the
    marginal log-likelihood as a NumPyro factor, allowing for downstream parameter inference.

    Examples:
        >>> def model(obs_times=None, obs_values=None):
        ...     dynamics = DynamicalModel(...)
        ...     return dsx.sample("f", dynamics, obs_times=obs_times, obs_values=obs_values)
        >>> def filtered_model(t, y):
        ...     with Filter(filter_config=KFConfig()):
        ...         return model(obs_times=t, obs_values=y)

    What this does
    --------------
    Filtering is the recursive (potentially approximate) computation of the filtering distribution
    \(p(x_t \mid y_{1:t})\). It allows for the computation of the marginal likelihood:

    \[
      \log p(y_{1:T}) = \sum_{t=1}^T \log p(y_t \mid y_{1:t-1}),
    \]

    which in turn can be used to compute the posterior distribution over the parameters $p(\theta | y_{1:T})$.


    Available Filter Configurations
    ----------------------------------
    There are several different filters available in `dynestyx`, each with their own strengths and weaknesses.
    What filters are applicable to a given model depends heavily on any special structure of the model (for example, linear and/or Gaussian observations).
    For a summary table of all config classes and when to use them, see
    [Available filter configurations](../filter_configs.md).

    Defaults
    --------
    If `filter_config=None`, defaults are:

    - `ContinuousTimeEnKFConfig()` for continuous-time models, and
    - `EKFConfig(filter_source="cuthbert")` for discrete-time models.

    Notes:
        - If your latent state is *discrete* (an HMM), you must use `HMMConfig`.
        - What gets recorded to the trace (means/covariances, particles/weights,
        etc.) depends on `filter_config.record_*` and the backend implementation.

    Attributes:
        filter_config: Selects the filtering algorithm and its hyperparameters.
            If `None`, a reasonable default is chosen based on whether the model
            is continuous-time or discrete-time.
    """

    filter_config: BaseFilterConfig | None = None

    def _add_log_factors(
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


def _filter_discrete_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> None:
    """Discrete-time marginal likelihood via cuthbert, pfjax, or cd-dynamax.

    Filter type inferred from config class: KFConfig, EKFConfig, UKFConfig (cd-dynamax)
    or EKFConfig (cuthbert), PFConfig (cuthbert or pfjax).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: Configuration for the filter.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).
    """

    if filter_config.filter_source == "cd_dynamax":
        run_cd_dynamax_discrete(
            name,
            dynamics,
            filter_config,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    elif filter_config.filter_source == "cuthbert":
        run_cuthbert_discrete(
            name,
            dynamics,
            filter_config,
            key=key,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    elif filter_config.filter_source == "pfjax":
        run_pfjax_discrete(
            name,
            dynamics,
            filter_config,
            key=key,
            obs_times=obs_times,
            obs_values=obs_values,
            ctrl_times=ctrl_times,
            ctrl_values=ctrl_values,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown filter source: {filter_config.filter_source}")


def _filter_continuous_time(
    name: str,
    dynamics: DynamicalModel,
    filter_config: BaseFilterConfig,
    key: jax.Array | None = None,
    *,
    obs_times: jax.Array,
    obs_values: jax.Array,
    ctrl_times=None,
    ctrl_values=None,
    **kwargs,
) -> None:
    """Continuous-time marginal likelihood via CD-Dynamax.

    Supports: EnKF, DPF, EKF, UKF (inferred from config type).

    Args:
        name: Name of the factor.
        dynamics: Dynamical model to filter.
        filter_config: Configuration for the filter.
        obs_times: Observation times.
        obs_values: Observed values.
        ctrl_times: Control times (optional).
        ctrl_values: Control values (optional).
    """
    run_continuous_filter(
        name,
        dynamics,
        filter_config,  # type: ignore[arg-type]
        key=key,
        obs_times=obs_times,
        obs_values=obs_values,
        ctrl_times=ctrl_times,
        ctrl_values=ctrl_values,
        **kwargs,
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
    "MarginalPFConfig",
    "PFConfig",
    "PFResamplingConfig",
    "UKFConfig",
]
