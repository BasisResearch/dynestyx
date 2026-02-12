from __future__ import annotations

import dataclasses
import hashlib
import itertools
from collections.abc import Iterable
from contextlib import ExitStack

import jax.random as jr
import pytest
from numpyro.handlers import seed
from numpyro.infer import Predictive

from dynestyx.filters import (
    FilterBasedHMMMarginalLogLikelihood,
    FilterBasedMarginalLogLikelihood,
)
from dynestyx.handlers import Condition, Discretizer
from dynestyx.simulators import DiscreteTimeSimulator, ODESimulator, SDESimulator
from tests.combinatorial.utils import (
    _build_context,
    _build_model,
    _build_predictive_context,
)

MODEL_KINDS = ["continuous", "discrete"]
STATE_EVOLUTION_KINDS = ["function", "module"]
INIT_KINDS = ["mvn", "uniform", "categorical"]
OBS_KINDS = ["linear_gaussian", "perfect", "poisson"]
DRIFT_KINDS = ["linear", "zero"]
DIFFUSION_KINDS = ["eye", "none"]
TRANSITION_KINDS = ["mvn_linear", "mvn_nonlinear", "categorical"]
HANDLERS = [
    "DiscreteTimeSimulator",
    "SDESimulator",
    "ODESimulator",
    "FilterHMM",
    "Filter",
]
FILTER_TYPES = ["EnKF", "EKF", "UKF", "DPF", "taylor_kf", "pf", "kf"]

DIMS = [1, 2]
TIMESTEPS = [1, 10]
USE_CONTROLS = [False, True]


@dataclasses.dataclass(frozen=True)
class ConditionedCase:
    case_id: str
    expected: str
    expected_error: str
    model_kind: str
    dim: int
    timesteps: int
    use_controls: bool
    control_dim: int
    init_kind: str
    state_evolution_kind: str
    drift_kind: str | None
    diffusion_coeff: str | None
    diffusion_cov: str | None
    transition_kind: str | None
    observation_kind: str
    handler: str
    filter_type: str | None
    use_discretizer: bool


@dataclasses.dataclass(frozen=True)
class PredictiveCase:
    case_id: str
    expected: str
    expected_error: str
    model_kind: str
    dim: int
    timesteps: int
    use_controls: bool
    control_dim: int
    init_kind: str
    state_evolution_kind: str
    drift_kind: str | None
    diffusion_coeff: str | None
    diffusion_cov: str | None
    transition_kind: str | None
    observation_kind: str
    use_discretizer: bool
    context_variant: str


def _case_id(parts: Iterable[str]) -> str:
    return "|".join(parts)


def _parametrize_case(case):
    marks = []
    if case.expected == "fail":
        marks.append(
            pytest.mark.xfail(
                reason=f"expected_error={case.expected_error}", strict=False
            )
        )
    return pytest.param(case, id=case.case_id, marks=marks)


def _maybe_thin_cases(cases, config):
    if not config.getoption("dsx_thin_combinatorial"):
        return cases
    limit = config.getoption("dsx_case_limit")
    if len(cases) <= limit:
        return cases

    def score(case):
        digest = hashlib.sha1(case.case_id.encode("utf-8")).hexdigest()
        return int(digest, 16)

    return [case for _, case in sorted((score(case), case) for case in cases)[:limit]]


def _expected_outcome_for_filter(case, effective_kind: str):
    filter_type = (case.filter_type or "default").lower()
    if effective_kind == "continuous":
        if filter_type not in ["default", "enkf", "ekf", "ukf", "dpf"]:
            return "fail", "ValueError: invalid filter_type"
        if case.init_kind in ["uniform", "categorical"]:
            return "fail", "NotImplementedError: initial_condition"
        if case.diffusion_coeff == "none" or case.diffusion_cov == "none":
            return "fail", "ValueError: missing diffusion"
        if filter_type in ["default", "enkf", "ekf", "ukf"]:
            if case.observation_kind != "linear_gaussian":
                return "fail", "TypeError: non_gaussian_obs"
        return "pass", "None"

    if filter_type not in ["default", "taylor_kf", "pf"]:
        return "fail", "ValueError: invalid filter_type"
    if case.timesteps == 1:
        return "fail", "IndexError: requires >=2 timesteps"
    if case.transition_kind == "categorical" and case.init_kind != "categorical":
        return "fail", "TypeError: init/transition mismatch"
    if case.transition_kind != "categorical" and case.init_kind == "categorical":
        return "fail", "TypeError: init/transition mismatch"
    if case.transition_kind == "categorical" and case.observation_kind in [
        "linear_gaussian",
        "perfect",
    ]:
        return "fail", "TypeError: categorical obs shape"
    if filter_type in ["default", "taylor_kf"]:
        if case.transition_kind == "categorical":
            return "fail", "ValueError: transition mean"
        if case.init_kind == "categorical":
            return "fail", "ValueError: initial mean"
    if case.model_kind == "continuous" and case.use_discretizer:
        if case.diffusion_coeff == "none" or case.diffusion_cov == "none":
            return "fail", "TypeError: discretizer diffusion"
    return "pass", "None"


def _expected_outcome_conditioned(case: ConditionedCase):
    effective_kind = (
        "discrete"
        if case.model_kind == "continuous" and case.use_discretizer
        else case.model_kind
    )
    if case.handler == "Filter":
        return _expected_outcome_for_filter(case, effective_kind)
    if case.handler == "FilterHMM":
        if effective_kind != "discrete":
            return "fail", "TypeError: hmm requires discrete"
        if case.transition_kind != "categorical":
            return "fail", "TypeError: hmm transition"
        if case.observation_kind in ["linear_gaussian", "perfect"]:
            return "fail", "TypeError: categorical obs shape"
        return "pass", "None"
    if case.handler == "SDESimulator":
        if effective_kind != "continuous":
            return "fail", "NotImplementedError: requires ContinuousTimeStateEvolution"
        if case.diffusion_coeff == "none" or case.diffusion_cov == "none":
            return "fail", "ValueError: requires diffusion"
        if case.init_kind in ["uniform", "categorical"]:
            return "fail", "NotImplementedError: initial_condition"
        return "pass", "None"
    if case.handler == "ODESimulator":
        if effective_kind != "continuous":
            return "fail", "AttributeError: requires ContinuousTimeStateEvolution"
        if case.init_kind == "categorical":
            return "fail", "TypeError: categorical init for ODE"
        return "pass", "None"
    if case.handler == "DiscreteTimeSimulator":
        if effective_kind != "discrete":
            return "fail", "TypeError: requires discrete evolution"
        if case.transition_kind == "categorical" and case.init_kind != "categorical":
            return "fail", "TypeError: init/transition mismatch"
        if case.transition_kind != "categorical" and case.init_kind == "categorical":
            return "fail", "TypeError: init/transition mismatch"
        if case.transition_kind == "categorical" and case.observation_kind in [
            "linear_gaussian",
            "perfect",
        ]:
            return "fail", "TypeError: categorical obs shape"
        if case.model_kind == "continuous" and case.use_discretizer:
            if case.diffusion_coeff == "none" or case.diffusion_cov == "none":
                return "fail", "TypeError: discretizer diffusion"
        return "pass", "None"
    return "pass", "None"


def _expected_outcome_predictive(case: PredictiveCase, handler: str):
    effective_kind = (
        "discrete"
        if case.model_kind == "continuous" and case.use_discretizer
        else case.model_kind
    )
    if handler == "DiscreteTimeSimulator":
        if effective_kind != "discrete":
            return "fail", "TypeError: requires discrete evolution"
        if case.transition_kind == "categorical" and case.init_kind != "categorical":
            return "fail", "TypeError: init/transition mismatch"
        if case.transition_kind != "categorical" and case.init_kind == "categorical":
            return "fail", "TypeError: init/transition mismatch"
        if case.transition_kind == "categorical" and case.observation_kind in [
            "linear_gaussian",
            "perfect",
        ]:
            return "fail", "TypeError: categorical obs shape"
        if case.model_kind == "continuous" and case.use_discretizer:
            if case.diffusion_coeff == "none" or case.diffusion_cov == "none":
                return "fail", "TypeError: discretizer diffusion"
        return "pass", "None"
    if handler == "SDESimulator":
        if effective_kind != "continuous":
            return "fail", "NotImplementedError: requires ContinuousTimeStateEvolution"
        if case.diffusion_coeff == "none" or case.diffusion_cov == "none":
            return "fail", "ValueError: requires diffusion"
        if case.init_kind in ["uniform", "categorical"]:
            return "fail", "NotImplementedError: initial_condition"
        return "pass", "None"
    if handler == "ODESimulator":
        if effective_kind != "continuous":
            return "fail", "AttributeError: requires ContinuousTimeStateEvolution"
        if case.init_kind == "categorical":
            return "fail", "TypeError: categorical init for ODE"
        return "pass", "None"
    return "pass", "None"


def _iter_conditioned_cases():
    cases = []
    for (
        model_kind,
        state_evolution_kind,
        init_kind,
        observation_kind,
        dim,
        timesteps,
        use_controls,
    ) in itertools.product(
        MODEL_KINDS,
        STATE_EVOLUTION_KINDS,
        INIT_KINDS,
        OBS_KINDS,
        DIMS,
        TIMESTEPS,
        USE_CONTROLS,
    ):
        control_dim = dim if use_controls else 0
        if model_kind == "continuous":
            for drift_kind, diffusion_coeff, diffusion_cov in itertools.product(
                DRIFT_KINDS, DIFFUSION_KINDS, DIFFUSION_KINDS
            ):
                for handler in HANDLERS:
                    for use_discretizer in [False, True]:
                        filter_types = FILTER_TYPES if handler == "Filter" else [None]
                        for filter_type in filter_types:
                            case_id = _case_id(
                                [
                                    "cond",
                                    "ct",
                                    f"dim={dim}",
                                    f"T={timesteps}",
                                    f"ctrl={'y' if use_controls else 'n'}",
                                    f"init={init_kind}",
                                    f"evo={state_evolution_kind}",
                                    f"drift={drift_kind}",
                                    f"diff={diffusion_coeff}/{diffusion_cov}",
                                    f"obs={observation_kind}",
                                    f"handler={handler}",
                                    f"filt={filter_type or 'na'}",
                                    f"disc={'y' if use_discretizer else 'n'}",
                                ]
                            )
                            temp_case = ConditionedCase(
                                case_id=case_id,
                                expected="pass",
                                expected_error="None",
                                model_kind=model_kind,
                                dim=dim,
                                timesteps=timesteps,
                                use_controls=use_controls,
                                control_dim=control_dim,
                                init_kind=init_kind,
                                state_evolution_kind=state_evolution_kind,
                                drift_kind=drift_kind,
                                diffusion_coeff=diffusion_coeff,
                                diffusion_cov=diffusion_cov,
                                transition_kind=None,
                                observation_kind=observation_kind,
                                handler=handler,
                                filter_type=filter_type,
                                use_discretizer=use_discretizer,
                            )
                            expected, expected_error = _expected_outcome_conditioned(
                                temp_case
                            )
                            cases.append(
                                dataclasses.replace(
                                    temp_case,
                                    expected=expected,
                                    expected_error=expected_error,
                                )
                            )
        else:
            for transition_kind in TRANSITION_KINDS:
                for handler in HANDLERS:
                    filter_types = FILTER_TYPES if handler == "Filter" else [None]
                    for filter_type in filter_types:
                        case_id = _case_id(
                            [
                                "cond",
                                "dt",
                                f"dim={dim}",
                                f"T={timesteps}",
                                f"ctrl={'y' if use_controls else 'n'}",
                                f"init={init_kind}",
                                f"evo={state_evolution_kind}",
                                f"trans={transition_kind}",
                                f"obs={observation_kind}",
                                f"handler={handler}",
                                f"filt={filter_type or 'na'}",
                                "disc=n",
                            ]
                        )
                        temp_case = ConditionedCase(
                            case_id=case_id,
                            expected="pass",
                            expected_error="None",
                            model_kind=model_kind,
                            dim=dim,
                            timesteps=timesteps,
                            use_controls=use_controls,
                            control_dim=control_dim,
                            init_kind=init_kind,
                            state_evolution_kind=state_evolution_kind,
                            drift_kind=None,
                            diffusion_coeff=None,
                            diffusion_cov=None,
                            transition_kind=transition_kind,
                            observation_kind=observation_kind,
                            handler=handler,
                            filter_type=filter_type,
                            use_discretizer=False,
                        )
                        expected, expected_error = _expected_outcome_conditioned(
                            temp_case
                        )
                        cases.append(
                            dataclasses.replace(
                                temp_case,
                                expected=expected,
                                expected_error=expected_error,
                            )
                        )
    return cases


def _select_predictive_handler(case: PredictiveCase) -> str:
    effective_kind = (
        "discrete"
        if case.model_kind == "continuous" and case.use_discretizer
        else case.model_kind
    )
    if effective_kind == "discrete":
        return "DiscreteTimeSimulator"
    if case.diffusion_coeff == "none" or case.diffusion_cov == "none":
        return "ODESimulator"
    return "SDESimulator"


def _iter_predictive_cases():
    cases = []
    for (
        model_kind,
        state_evolution_kind,
        init_kind,
        observation_kind,
        dim,
        timesteps,
        use_controls,
    ) in itertools.product(
        MODEL_KINDS,
        STATE_EVOLUTION_KINDS,
        INIT_KINDS,
        OBS_KINDS,
        DIMS,
        TIMESTEPS,
        USE_CONTROLS,
    ):
        control_dim = dim if use_controls else 0
        context_variants = [
            "obs_times",
            "obs_times_values",
            "obs_times_values_controls",
        ]
        if model_kind == "continuous":
            for drift_kind, diffusion_coeff, diffusion_cov in itertools.product(
                DRIFT_KINDS, DIFFUSION_KINDS, DIFFUSION_KINDS
            ):
                for use_discretizer in [False, True]:
                    for context_variant in context_variants:
                        case_id = _case_id(
                            [
                                "pred",
                                "ct",
                                f"dim={dim}",
                                f"T={timesteps}",
                                f"ctrl={'y' if use_controls else 'n'}",
                                f"init={init_kind}",
                                f"evo={state_evolution_kind}",
                                f"drift={drift_kind}",
                                f"diff={diffusion_coeff}/{diffusion_cov}",
                                f"obs={observation_kind}",
                                f"ctx={context_variant}",
                                f"disc={'y' if use_discretizer else 'n'}",
                            ]
                        )
                        temp_case = PredictiveCase(
                            case_id=case_id,
                            expected="pass",
                            expected_error="None",
                            model_kind=model_kind,
                            dim=dim,
                            timesteps=timesteps,
                            use_controls=use_controls,
                            control_dim=control_dim,
                            init_kind=init_kind,
                            state_evolution_kind=state_evolution_kind,
                            drift_kind=drift_kind,
                            diffusion_coeff=diffusion_coeff,
                            diffusion_cov=diffusion_cov,
                            transition_kind=None,
                            observation_kind=observation_kind,
                            use_discretizer=use_discretizer,
                            context_variant=context_variant,
                        )
                        handler = _select_predictive_handler(temp_case)
                        expected, expected_error = _expected_outcome_predictive(
                            temp_case, handler
                        )
                        cases.append(
                            dataclasses.replace(
                                temp_case,
                                expected=expected,
                                expected_error=expected_error,
                            )
                        )
        else:
            for transition_kind in TRANSITION_KINDS:
                for context_variant in context_variants:
                    case_id = _case_id(
                        [
                            "pred",
                            "dt",
                            f"dim={dim}",
                            f"T={timesteps}",
                            f"ctrl={'y' if use_controls else 'n'}",
                            f"init={init_kind}",
                            f"evo={state_evolution_kind}",
                            f"trans={transition_kind}",
                            f"obs={observation_kind}",
                            f"ctx={context_variant}",
                            "disc=n",
                        ]
                    )
                    temp_case = PredictiveCase(
                        case_id=case_id,
                        expected="pass",
                        expected_error="None",
                        model_kind=model_kind,
                        dim=dim,
                        timesteps=timesteps,
                        use_controls=use_controls,
                        control_dim=control_dim,
                        init_kind=init_kind,
                        state_evolution_kind=state_evolution_kind,
                        drift_kind=None,
                        diffusion_coeff=None,
                        diffusion_cov=None,
                        transition_kind=transition_kind,
                        observation_kind=observation_kind,
                        use_discretizer=False,
                        context_variant=context_variant,
                    )
                    handler = _select_predictive_handler(temp_case)
                    expected, expected_error = _expected_outcome_predictive(
                        temp_case, handler
                    )
                    cases.append(
                        dataclasses.replace(
                            temp_case, expected=expected, expected_error=expected_error
                        )
                    )
    return cases


def pytest_generate_tests(metafunc):
    if "conditioned_case" in metafunc.fixturenames:
        cases = _iter_conditioned_cases()
        cases = _maybe_thin_cases(cases, metafunc.config)
        metafunc.parametrize(
            "conditioned_case", [_parametrize_case(case) for case in cases]
        )
    if "predictive_case" in metafunc.fixturenames:
        cases = _iter_predictive_cases()
        cases = _maybe_thin_cases(cases, metafunc.config)
        metafunc.parametrize(
            "predictive_case", [_parametrize_case(case) for case in cases]
        )


def _run_with_handlers(handlers, run_fn, rng_key):
    with ExitStack() as stack:
        stack.enter_context(seed(rng_seed=rng_key))
        for handler in handlers:
            stack.enter_context(handler)
        run_fn()


def test_conditioned_forward_pass(conditioned_case: ConditionedCase, request):
    request.node._dynestyx_case = conditioned_case

    model = _build_model(
        model_kind=conditioned_case.model_kind,
        dim=conditioned_case.dim,
        control_dim=conditioned_case.control_dim,
        init_kind=conditioned_case.init_kind,
        state_evolution_kind=conditioned_case.state_evolution_kind,
        drift_kind=conditioned_case.drift_kind,
        diffusion_coeff=conditioned_case.diffusion_coeff,
        diffusion_cov=conditioned_case.diffusion_cov,
        transition_kind=conditioned_case.transition_kind,
        observation_kind=conditioned_case.observation_kind,
    )
    context = _build_context(
        observation_kind=conditioned_case.observation_kind,
        timesteps=conditioned_case.timesteps,
        dim=conditioned_case.dim,
        use_controls=conditioned_case.use_controls,
        control_dim=conditioned_case.control_dim,
    )

    handlers = []
    if conditioned_case.handler == "DiscreteTimeSimulator":
        handlers.append(DiscreteTimeSimulator())
    elif conditioned_case.handler == "SDESimulator":
        handlers.append(SDESimulator(key=jr.PRNGKey(0)))
    elif conditioned_case.handler == "ODESimulator":
        handlers.append(ODESimulator())
    elif conditioned_case.handler == "Filter":
        filter_kwargs = {}
        if (conditioned_case.filter_type or "").lower() == "pf":
            filter_kwargs["n_filter_particles"] = 5
        if (conditioned_case.filter_type or "").lower() == "dpf":
            filter_kwargs["N_particles"] = 5
        handlers.append(
            FilterBasedMarginalLogLikelihood(
                filter_type=conditioned_case.filter_type or "default",
                **filter_kwargs,
            )
        )
    elif conditioned_case.handler == "FilterHMM":
        handlers.append(FilterBasedHMMMarginalLogLikelihood())

    if conditioned_case.use_discretizer:
        handlers.append(Discretizer())
    handlers.append(Condition(context))

    _run_with_handlers(handlers, model, jr.PRNGKey(0))


def test_predictive_forward_pass(predictive_case: PredictiveCase, request):
    request.node._dynestyx_case = predictive_case

    model = _build_model(
        model_kind=predictive_case.model_kind,
        dim=predictive_case.dim,
        control_dim=predictive_case.control_dim,
        init_kind=predictive_case.init_kind,
        state_evolution_kind=predictive_case.state_evolution_kind,
        drift_kind=predictive_case.drift_kind,
        diffusion_coeff=predictive_case.diffusion_coeff,
        diffusion_cov=predictive_case.diffusion_cov,
        transition_kind=predictive_case.transition_kind,
        observation_kind=predictive_case.observation_kind,
    )
    context = _build_predictive_context(
        observation_kind=predictive_case.observation_kind,
        timesteps=predictive_case.timesteps,
        dim=predictive_case.dim,
        use_controls=predictive_case.use_controls,
        control_dim=predictive_case.control_dim,
        context_variant=predictive_case.context_variant,
    )

    handlers = []
    handler_name = _select_predictive_handler(predictive_case)
    if handler_name == "DiscreteTimeSimulator":
        handlers.append(DiscreteTimeSimulator())
    elif handler_name == "SDESimulator":
        handlers.append(SDESimulator(key=jr.PRNGKey(1)))
    else:
        handlers.append(ODESimulator())

    if predictive_case.use_discretizer:
        handlers.append(Discretizer())
    handlers.append(Condition(context))

    predictive = Predictive(model, num_samples=1, exclude_deterministic=False)

    def run():
        predictive(jr.PRNGKey(0))

    _run_with_handlers(handlers, run, jr.PRNGKey(0))
