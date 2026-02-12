from dataclasses import dataclass


@dataclass(frozen=True)
class DataSpec:
    obs_rank: int  # 1 -> (T,), 2 -> (T, 2)
    timesteps: int
    ctrl_rank: int  # 0 -> none, 1 -> (T,), 2 -> (T,2)


@dataclass(frozen=True)
class ModelSpec:
    family: str  # "discrete", "continuous"
    discrete_kind: str | None  # discrete only: "gaussian", "categorical_hmm"
    initial_kind: str  # mvn, uniform, categorical
    init_rank: int  # 1 or 2
    uses_control: bool
    transition_kind: str  # linear_mvn, nonlinear_mvn, categorical, linear_ct, zero_ct
    diffusion_coeff: str  # eye, none (continuous only)
    diffusion_cov: str  # eye, none (continuous only)
    observation_kind: str  # linear_gaussian, perfect, poisson
    observation_rank: int  # 1 or 2


@dataclass(frozen=True)
class InferenceSpec:
    runner: str  # discrete_sim, ode_sim, sde_sim, filter, filter_hmm
    filter_type: str | None
    discretizer: str  # none, default, euler


@dataclass
class CaseResult:
    case_id: str
    timesteps: int
    control_rank: int
    expected_pass: bool
    expected_error: str | None
    actual_pass: bool
    actual_error: str | None
    model: ModelSpec
    inference: InferenceSpec


@dataclass
class PredictiveCaseResult:
    case_id: str
    source_case_id: str
    timesteps: int
    control_rank: int
    model_family: str
    model_discrete_kind: str | None
    initial_kind: str
    init_rank: int
    observation_kind: str
    observation_rank: int
    transition_kind: str
    runner: str
    filter_type: str | None
    discretizer: str
    context_mode: str
    expected_pass: bool
    expected_error: str | None
    actual_pass: bool
    actual_error: str | None

