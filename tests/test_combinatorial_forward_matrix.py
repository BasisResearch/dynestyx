import itertools
import sys

from tests.combinatorial.assumptions import (
    expected_outcome,
)
from tests.combinatorial.config import (
    PREDICTIVE_SCORECARD_MISMATCH_XLS_PATH,
    PREDICTIVE_SCORECARD_PATH,
    PREDICTIVE_SCORECARD_XLS_PATH,
    SCORECARD_MISMATCH_XLS_PATH,
    SCORECARD_PATH,
    SCORECARD_XLS_PATH,
)
from tests.combinatorial.execution import run_forward_case, run_predictive_case
from tests.combinatorial.matrix import (
    build_forward_model_specs,
    build_inference_specs,
    predictive_context_modes,
    sample_cases,
)
from tests.combinatorial.model_factory import make_data
from tests.combinatorial.reporting import (
    case_results_to_rows,
    predictive_results_to_rows,
    render_table,
    write_case_results_json,
    write_predictive_results_json,
    write_xls_scorecard,
)
from tests.combinatorial.specs import (
    CaseResult,
    DataSpec,
    PredictiveCaseResult,
)

_FORWARD_RESULTS_CACHE: dict[tuple[bool, int], list[CaseResult]] = {}


def _progress_text(prefix: str, done: int, total: int, width: int = 24) -> str:
    total = max(total, 1)
    done = min(max(done, 0), total)
    frac = done / total
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    return f"{prefix}: [{bar}] {done}/{total} ({frac * 100:5.1f}%)"


def _emit_progress(prefix: str, done: int, total: int):
    text = _progress_text(prefix=prefix, done=done, total=total)
    end = "\n" if done >= total else "\r"
    print(text, end=end, flush=True)
    if end == "\r":
        # Keep terminal output tidy when tools buffer carriage returns.
        sys.stdout.flush()


def _compute_forward_results(
    thin: bool,
    case_limit: int,
    *,
    show_progress: bool = False,
    progress_every: int = 25,
) -> list[CaseResult]:
    cache_key = (thin, case_limit)
    if cache_key in _FORWARD_RESULTS_CACHE:
        return _FORWARD_RESULTS_CACHE[cache_key]

    matrix = itertools.product(build_forward_model_specs(), build_inference_specs())
    selected_cases = sample_cases(case_limit=case_limit, combos=matrix, thin=thin)
    total = len(selected_cases) * 2
    progress_every = max(1, progress_every)

    results: list[CaseResult] = []
    case_idx = 0
    for model_spec, inference_spec in selected_cases:
        for timesteps in (1, 3):
            case_id = f"case_{case_idx:04d}"
            case_idx += 1
            expected_pass, expected_error = expected_outcome(model_spec, inference_spec)
            actual_pass = True
            actual_error = None
            try:
                run_forward_case(model_spec, inference_spec, timesteps=timesteps)
            except Exception as exc:  # noqa: BLE001 - scorecard records all failures.
                actual_pass = False
                actual_error = type(exc).__name__

            results.append(
                CaseResult(
                    case_id=case_id,
                    timesteps=timesteps,
                    control_rank=2 if model_spec.uses_control else 0,
                    expected_pass=expected_pass,
                    expected_error=expected_error,
                    actual_pass=actual_pass,
                    actual_error=actual_error,
                    model=model_spec,
                    inference=inference_spec,
                )
            )
            if show_progress and (
                case_idx == 1 or case_idx % progress_every == 0 or case_idx == total
            ):
                _emit_progress("Forward cases", case_idx, total)

    _FORWARD_RESULTS_CACHE[cache_key] = results
    return results


def test_data_generation_matrix():
    data_specs = [
        DataSpec(obs_rank=o, timesteps=t, ctrl_rank=c)
        for o, t, c in itertools.product([1, 2], [1, 3], [0, 1, 2])
    ]
    for spec in data_specs:
        times, obs_values, controls, _ = make_data(spec)
        assert times.shape[0] == spec.timesteps
        if spec.obs_rank == 1:
            assert obs_values.shape == (spec.timesteps,)
        else:
            assert obs_values.shape == (spec.timesteps, 2)
        if spec.ctrl_rank == 0:
            assert controls.values is None
        elif spec.ctrl_rank == 1:
            assert controls.values.shape == (spec.timesteps,)
        else:
            assert controls.values.shape == (spec.timesteps, 2)


def test_forward_pass_combinatorial_scorecard(capsys, request):
    thin = bool(request.config.getoption("--dsx-thin-combinatorial"))
    case_limit = int(request.config.getoption("--dsx-case-limit"))
    show_progress = bool(request.config.getoption("dsx_progress"))
    progress_every = int(request.config.getoption("--dsx-progress-every"))
    with capsys.disabled():
        results = _compute_forward_results(
            thin=thin,
            case_limit=case_limit,
            show_progress=show_progress,
            progress_every=progress_every,
        )

    write_case_results_json(SCORECARD_PATH, results)
    xls_headers, xls_rows, xls_mismatches = case_results_to_rows(results)
    write_xls_scorecard(
        path=SCORECARD_XLS_PATH,
        sheet_name="Combinatorial",
        headers=xls_headers,
        rows=xls_rows,
        mismatch_flags=xls_mismatches,
    )
    mismatch_rows = [row for row, is_mm in zip(xls_rows, xls_mismatches, strict=True) if is_mm]
    mismatch_flags = [True] * len(mismatch_rows)
    write_xls_scorecard(
        path=SCORECARD_MISMATCH_XLS_PATH,
        sheet_name="CombinatorialMismatches",
        headers=xls_headers,
        rows=mismatch_rows,
        mismatch_flags=mismatch_flags,
    )

    mismatch_count = sum(1 for r in results if r.expected_pass != r.actual_pass)
    summary_line = (
        "Combinatorial scorecard summary: "
        f"cases={len(results)} "
        f"expected_pass={sum(r.expected_pass for r in results)} "
        f"actual_pass={sum(r.actual_pass for r in results)} "
        f"mismatches={mismatch_count}"
    )
    with capsys.disabled():
        print(summary_line)
        print("Combinatorial scorecard table:")
        print(render_table(xls_headers, xls_rows))

    assert results
    assert any(r.actual_pass for r in results)
    assert any(not r.actual_pass for r in results)


def test_predictive_context_matrix(capsys, request):
    results: list[PredictiveCaseResult] = []
    # Predictive grid is now driven by the forward scorecard:
    # run Predictive for every case that was expected-pass AND actual-pass there.
    thin = bool(request.config.getoption("--dsx-thin-combinatorial"))
    case_limit = int(request.config.getoption("--dsx-case-limit"))
    show_progress = bool(request.config.getoption("dsx_progress"))
    progress_every = max(1, int(request.config.getoption("--dsx-progress-every")))
    forward_results = _compute_forward_results(thin=thin, case_limit=case_limit)
    passing_forward_cases = [
        r for r in forward_results if r.expected_pass and r.actual_pass
    ]

    context_modes = predictive_context_modes()
    total_predictive = len(passing_forward_cases) * len(context_modes)
    case_idx = 0
    for forward_case in passing_forward_cases:
        model_spec = forward_case.model
        inf_spec = forward_case.inference
        for context_mode in context_modes:
            expected_pass, expected_error = expected_outcome(model_spec, inf_spec)
            actual_pass = True
            actual_error = None
            try:
                run_predictive_case(
                    model_spec,
                    inf_spec,
                    context_mode,
                    timesteps=forward_case.timesteps,
                )
            except Exception as exc:  # noqa: BLE001 - scorecard records all failures.
                actual_pass = False
                actual_error = type(exc).__name__

            results.append(
                PredictiveCaseResult(
                    case_id=f"predictive_{case_idx:03d}",
                    source_case_id=forward_case.case_id,
                    timesteps=forward_case.timesteps,
                    control_rank=2 if model_spec.uses_control else 0,
                    model_family=model_spec.family,
                    model_discrete_kind=model_spec.discrete_kind,
                    initial_kind=model_spec.initial_kind,
                    init_rank=model_spec.init_rank,
                    observation_kind=model_spec.observation_kind,
                    observation_rank=model_spec.observation_rank,
                    transition_kind=model_spec.transition_kind,
                    runner=inf_spec.runner,
                    filter_type=inf_spec.filter_type,
                    discretizer=inf_spec.discretizer,
                    context_mode=context_mode,
                    expected_pass=expected_pass,
                    expected_error=expected_error,
                    actual_pass=actual_pass,
                    actual_error=actual_error,
                )
            )
            case_idx += 1
            if show_progress and (
                case_idx == 1
                or case_idx % progress_every == 0
                or case_idx == total_predictive
            ):
                with capsys.disabled():
                    _emit_progress("Predictive cases", case_idx, total_predictive)

    write_predictive_results_json(PREDICTIVE_SCORECARD_PATH, results)
    xls_headers, xls_rows, xls_mismatches = predictive_results_to_rows(results)
    write_xls_scorecard(
        path=PREDICTIVE_SCORECARD_XLS_PATH,
        sheet_name="Predictive",
        headers=xls_headers,
        rows=xls_rows,
        mismatch_flags=xls_mismatches,
    )
    mismatch_rows = [row for row, is_mm in zip(xls_rows, xls_mismatches, strict=True) if is_mm]
    mismatch_flags = [True] * len(mismatch_rows)
    write_xls_scorecard(
        path=PREDICTIVE_SCORECARD_MISMATCH_XLS_PATH,
        sheet_name="PredictiveMismatches",
        headers=xls_headers,
        rows=mismatch_rows,
        mismatch_flags=mismatch_flags,
    )

    context_count = len(predictive_context_modes())
    actual_pass_count = sum(r.actual_pass for r in results)
    actual_fail_count = len(results) - actual_pass_count
    mismatch_count = sum(1 for r in results if r.expected_pass != r.actual_pass)
    summary_line = (
        "Predictive scorecard summary: "
        f"forward_passpass_cases={len(passing_forward_cases)} "
        f"context_modes={context_count} "
        f"cases={len(results)} "
        f"expected_pass={sum(r.expected_pass for r in results)} "
        f"actual_pass={actual_pass_count} "
        f"actual_fail={actual_fail_count} "
        f"mismatches={mismatch_count}"
    )
    failure_results = [r for r in results if not r.actual_pass]
    _, failure_rows, _ = predictive_results_to_rows(failure_results)
    with capsys.disabled():
        print(summary_line)
        print("Predictive failures table:")
        if failure_rows:
            print(render_table(xls_headers, failure_rows))
        else:
            print("(no predictive failures)")

    assert results
    assert any(r.actual_pass for r in results)
    # Predictive cases are now sourced from forward pass-pass cases, so all-pass is acceptable.

