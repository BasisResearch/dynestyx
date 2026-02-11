import itertools

from tests.combinatorial.config import THIN_COMBINATORIAL
from tests.combinatorial.specs import InferenceSpec, ModelSpec


def sample_cases(case_limit: int, combos, thin: bool | None = None):
    use_thin = THIN_COMBINATORIAL if thin is None else thin
    if not use_thin:
        return list(combos)
    combo_list = list(combos)
    if len(combo_list) <= case_limit:
        return combo_list

    selected_indices: list[int] = []
    seen_indices: set[int] = set()

    # Ensure all filter types appear at least once in sampled output.
    required_filter_types = ["EnKF", "EKF", "UKF", "DPF", "taylor_kf", "pf", "kf"]
    for filter_type in required_filter_types:
        for idx, combo in enumerate(combo_list):
            if not isinstance(combo, tuple) or len(combo) < 2:
                continue
            inf_spec = combo[1]
            if getattr(inf_spec, "filter_type", None) == filter_type:
                if idx not in seen_indices:
                    selected_indices.append(idx)
                    seen_indices.add(idx)
                break

    # Fill remaining slots with deterministic striding.
    stride = max(1, len(combo_list) // case_limit)
    for idx in range(0, len(combo_list), stride):
        if len(selected_indices) >= case_limit:
            break
        if idx in seen_indices:
            continue
        selected_indices.append(idx)
        seen_indices.add(idx)

    # If stride fill leaves a gap, top up sequentially.
    if len(selected_indices) < case_limit:
        for idx in range(len(combo_list)):
            if len(selected_indices) >= case_limit:
                break
            if idx in seen_indices:
                continue
            selected_indices.append(idx)
            seen_indices.add(idx)

    return [combo_list[idx] for idx in selected_indices[:case_limit]]


def build_forward_model_specs() -> list[ModelSpec]:
    model_specs = [
        ModelSpec("discrete_gaussian", ik, ir, uc, tk, "none", "none", ok, orank)
        for ik, ir, uc, tk, ok, orank in itertools.product(
            ["mvn", "uniform", "categorical"],
            [1, 2],
            [False, True],
            ["linear_mvn", "nonlinear_mvn", "categorical"],
            ["linear_gaussian", "perfect", "poisson"],
            [1, 2],
        )
    ] + [
        ModelSpec("continuous", ik, ir, uc, tk, dc, dv, ok, orank)
        for ik, ir, uc, tk, dc, dv, ok, orank in itertools.product(
            ["mvn", "uniform", "categorical"],
            [1, 2],
            [False, True],
            ["zero_ct"],
            ["eye", "none"],
            ["eye", "none"],
            ["linear_gaussian", "perfect", "poisson"],
            [1, 2],
        )
    ]

    hmm_specs = [
        ModelSpec(
            "categorical_hmm",
            "categorical",
            ir,
            uc,
            "categorical",
            "none",
            "none",
            ok,
            orank,
        )
        for ir, uc, ok, orank in itertools.product(
            [1, 2], [False, True], ["poisson", "perfect"], [1, 2]
        )
    ]
    model_specs.extend(hmm_specs)
    return model_specs


def build_inference_specs() -> list[InferenceSpec]:
    return [
        InferenceSpec("discrete_sim", None, "none"),
        InferenceSpec("ode_sim", None, "none"),
        InferenceSpec("sde_sim", None, "none"),
        InferenceSpec("filter_hmm", None, "none"),
    ] + [
        InferenceSpec("filter", ftype, disc)
        for ftype, disc in itertools.product(
            ["EnKF", "EKF", "UKF", "DPF", "taylor_kf", "pf", "kf"],
            ["none", "euler"],
        )
    ]


def build_predictive_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            "discrete_gaussian",
            "mvn",
            2,
            True,
            "linear_mvn",
            "none",
            "none",
            "linear_gaussian",
            2,
        ),
        ModelSpec(
            "continuous",
            "mvn",
            2,
            True,
            "zero_ct",
            "eye",
            "eye",
            "linear_gaussian",
            2,
        ),
        ModelSpec(
            "categorical_hmm",
            "categorical",
            1,
            False,
            "categorical",
            "none",
            "none",
            "poisson",
            1,
        ),
    ]


def predictive_context_modes() -> list[str]:
    return [
        "obs_times",
        "obs_times_obs_values",
        "obs_times_obs_values_ctrl_times_ctrl_values",
    ]

