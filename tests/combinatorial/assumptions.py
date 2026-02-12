"""Centralized expected pass/fail assumptions for combinatorial tests."""

from tests.combinatorial.specs import InferenceSpec, ModelSpec


def is_discrete_filter(filter_type: str | None) -> bool:
    return filter_type is not None and filter_type.lower() in {"default", "taylor_kf", "pf"}


def is_continuous_filter(filter_type: str | None) -> bool:
    return filter_type is not None and filter_type.lower() in {
        "default",
        "enkf",
        "ekf",
        "ukf",
        "dpf",
    }


def expected_outcome(model: ModelSpec, inf: InferenceSpec) -> tuple[bool, str | None]:
    """Expected pass/fail + expected error tag for each forward-pass case."""
    is_discrete = model.family == "discrete"
    is_hmm_discrete = is_discrete and model.discrete_kind == "categorical_hmm"

    transition_is_categorical = model.transition_kind == "categorical"
    ic_is_categorical = model.initial_kind == "categorical"
    if ic_is_categorical and not transition_is_categorical:
        return False, "categorical_ic_requires_categorical_transition"
    if transition_is_categorical and not ic_is_categorical:
        return False, "categorical_transition_requires_categorical_ic"

    if is_hmm_discrete and inf.runner != "filter_hmm":
        return False, "hmm_filter_required"
    if model.observation_kind == "perfect" and model.observation_rank != model.init_rank:
        return False, "shape_mismatch"

    if inf.runner == "filter_hmm":
        ok = is_hmm_discrete
        return ok, None if ok else "hmm_required"

    if inf.runner == "sde_sim":
        if model.family != "continuous":
            return False, "continuous_required"
        if model.diffusion_coeff != "eye" or model.diffusion_cov != "eye":
            return False, "missing_diffusion"
        return True, None

    if inf.runner == "ode_sim":
        if model.family != "continuous":
            return False, "continuous_required"
        has_drift = model.transition_kind.endswith("_ct")
        if not has_drift:
            return False, "missing_drift"
        if model.diffusion_coeff != "none" or model.diffusion_cov != "none":
            return False, "ode_requires_no_diffusion"
        return True, None

    if inf.runner == "discrete_sim":
        if model.family == "continuous" and inf.discretizer == "none":
            return False, "not_discretized"
        return True, None

    if inf.runner == "filter":
        if model.family == "continuous" and inf.discretizer == "none":
            if is_continuous_filter(inf.filter_type):
                return True, None
            return False, "invalid_filter_type"

        if is_discrete_filter(inf.filter_type):
            return True, None
        return False, "invalid_filter_type"

    return False, "unsupported_runner"


def expected_predictive_outcome(model_family: str) -> tuple[bool, str | None]:
    """Expected pass/fail for Predictive-context matrix."""
    if model_family == "continuous":
        return False, "predictive_continuous_unsupported"
    return True, None

