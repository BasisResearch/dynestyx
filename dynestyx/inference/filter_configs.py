"""Filter configuration dataclasses. Shared by dispatchers and integration backends."""

import dataclasses
import math
from typing import Literal

ResamplingBaseMethod = Literal["systematic", "multinomial", "stratified"]
ResamplingDifferentiableMethod = Literal["stop_gradient", "straight_through", "soft"]
FilterSource = Literal["cuthbert", "cd_dynamax", "dynestyx"]
FilterEmissionOrder = Literal["zeroth", "first", "second"]
FilterStateOrder = Literal["zeroth", "first", "second"]


@dataclasses.dataclass
class BaseFilterConfig:
    extra_filter_kwargs: dict = dataclasses.field(default_factory=dict)
    warn: bool = True
    record_filtered_states_mean: bool | None = None
    record_filtered_states_cov: bool | None = None
    record_filtered_states_cov_diag: bool | None = None
    record_filtered_particles: bool | None = None
    record_filtered_log_weights: bool | None = None
    record_filtered_states_chol_cov: bool | None = None
    record_max_elems: int = 100_000
    filter_source: FilterSource | None = None
    cov_rescaling: float | None = None


@dataclasses.dataclass
class EnKFConfig(BaseFilterConfig):
    n_particles: int = 100
    perturb_measurements: bool | None = None
    inflation_delta: float | None = None
    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class PFResamplingConfig:
    base_method: ResamplingBaseMethod = "systematic"
    differential_method: ResamplingDifferentiableMethod = "stop_gradient"
    softness: float = 0.7


@dataclasses.dataclass
class PFConfig(BaseFilterConfig):
    n_particles: int = 100
    resampling_method: PFResamplingConfig = dataclasses.field(
        default_factory=PFResamplingConfig
    )
    ess_threshold_ratio: float = 0.7
    filter_source: FilterSource = "cuthbert"


@dataclasses.dataclass
class EKFConfig(BaseFilterConfig):
    filter_source: FilterSource = "cuthbert"
    filter_emission_order: FilterEmissionOrder = "first"


@dataclasses.dataclass
class KFConfig(BaseFilterConfig):
    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class UKFConfig(BaseFilterConfig):
    alpha: float = math.sqrt(3)
    beta: int = 2
    kappa: int = 1
    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class ContinuousTimeConfig:
    filter_state_order: FilterStateOrder = "first"
    diffeqsolve_max_steps: int = 1_000
    diffeqsolve_dt0: float = 0.01
    diffeqsolve_kwargs: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ContinuousTimeEnKFConfig(EnKFConfig, ContinuousTimeConfig):
    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class ContinuousTimeDPFConfig(PFConfig, ContinuousTimeConfig):
    filter_source: FilterSource = "cd_dynamax"
    resampling_method: PFResamplingConfig = dataclasses.field(
        default_factory=lambda: PFResamplingConfig(base_method="multinomial")
    )


@dataclasses.dataclass
class ContinuousTimeEKFConfig(EKFConfig, ContinuousTimeConfig):
    filter_source: FilterSource = "cd_dynamax"


@dataclasses.dataclass
class ContinuousTimeUKFConfig(UKFConfig, ContinuousTimeConfig):
    filter_source: FilterSource = "cd_dynamax"


DiscreteTimeConfigs: tuple[type, ...] = (
    EnKFConfig,
    PFConfig,
    EKFConfig,
    KFConfig,
    UKFConfig,
)

ContinuousTimeConfigs: tuple[type, ...] = (
    ContinuousTimeEnKFConfig,
    ContinuousTimeDPFConfig,
    ContinuousTimeEKFConfig,
    ContinuousTimeUKFConfig,
)


@dataclasses.dataclass
class HMMConfig(BaseFilterConfig):
    record_filtered: bool | None = None
    record_log_filtered: bool | None = None
    filter_source: FilterSource = "dynestyx"


HMMConfigs: tuple[type, ...] = (HMMConfig,)


def config_to_record_kwargs(config: BaseFilterConfig) -> dict:
    """Build record_kwargs dict from config. Config must have all record_* fields."""
    if isinstance(config, HMMConfig):
        return {
            "record_filtered": config.record_filtered,
            "record_log_filtered": config.record_log_filtered,
            "record_max_elems": config.record_max_elems,
        }
    else:
        return {
            "record_filtered_states_mean": config.record_filtered_states_mean,
            "record_filtered_states_cov": config.record_filtered_states_cov,
            "record_filtered_states_cov_diag": config.record_filtered_states_cov_diag,
            "record_filtered_particles": config.record_filtered_particles,
            "record_filtered_log_weights": config.record_filtered_log_weights,
            "record_filtered_states_chol_cov": config.record_filtered_states_chol_cov,
            "record_max_elems": config.record_max_elems,
        }
