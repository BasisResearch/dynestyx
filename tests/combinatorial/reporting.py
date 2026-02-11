import json
import os
from dataclasses import asdict
from xml.sax.saxutils import escape as xml_escape

from tests.combinatorial.config import (
    ANSI_BOLD,
    ANSI_CYAN,
    ANSI_DIM,
    ANSI_GREEN,
    ANSI_RED,
    ANSI_RESET,
    ANSI_YELLOW,
    USE_COLOR,
)
from tests.combinatorial.specs import CaseResult, PredictiveCaseResult


def _paint(text: str, color: str) -> str:
    if not USE_COLOR:
        return text
    return f"{color}{text}{ANSI_RESET}"


def _status_text(passed: bool, width: int) -> str:
    label = "pass" if passed else "fail"
    return _paint(label.ljust(width), ANSI_GREEN if passed else ANSI_RED)


def _flag_text(mismatch: bool, width: int) -> str:
    if mismatch:
        return _paint("MISMATCH".ljust(width), ANSI_YELLOW + ANSI_BOLD)
    return _paint("ok".ljust(width), ANSI_DIM)


def write_json_scorecard(path: str, payload: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_case_results_json(path: str, results: list[CaseResult]):
    write_json_scorecard(path, [asdict(r) for r in results])


def write_predictive_results_json(path: str, results: list[PredictiveCaseResult]):
    write_json_scorecard(path, [asdict(r) for r in results])


def write_xls_scorecard(
    path: str,
    sheet_name: str,
    headers: list[str],
    rows: list[list[str]],
    mismatch_flags: list[bool],
    expected_col: str = "expected",
    actual_col: str = "actual",
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    expected_idx = headers.index(expected_col)
    actual_idx = headers.index(actual_col)

    xml_lines = [
        '<?xml version="1.0"?>',
        '<?mso-application progid="Excel.Sheet"?>',
        '<Workbook xmlns="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:x="urn:schemas-microsoft-com:office:excel" '
        'xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet" '
        'xmlns:html="http://www.w3.org/TR/REC-html40">',
        "<Styles>",
        '<Style ss:ID="header"><Font ss:Bold="1" ss:Color="#FFFFFF"/><Interior ss:Color="#1F4E78" ss:Pattern="Solid"/></Style>',
        '<Style ss:ID="normal"/>',
        '<Style ss:ID="mismatch"><Interior ss:Color="#FFF2CC" ss:Pattern="Solid"/></Style>',
        '<Style ss:ID="pass"><Font ss:Color="#006100"/><Interior ss:Color="#C6EFCE" ss:Pattern="Solid"/></Style>',
        '<Style ss:ID="fail"><Font ss:Color="#9C0006"/><Interior ss:Color="#FFC7CE" ss:Pattern="Solid"/></Style>',
        "</Styles>",
        f'<Worksheet ss:Name="{xml_escape(sheet_name)}">',
        "<Table>",
        "<Row>",
    ]
    for h in headers:
        xml_lines.append(
            f'<Cell ss:StyleID="header"><Data ss:Type="String">{xml_escape(h)}</Data></Cell>'
        )
    xml_lines.append("</Row>")

    for row, is_mismatch in zip(rows, mismatch_flags, strict=True):
        xml_lines.append("<Row>")
        for col_idx, cell in enumerate(row):
            style = "mismatch" if is_mismatch else "normal"
            if col_idx in {expected_idx, actual_idx}:
                style = "pass" if str(cell).strip().lower() == "pass" else "fail"
            xml_lines.append(
                f'<Cell ss:StyleID="{style}"><Data ss:Type="String">{xml_escape(str(cell))}</Data></Cell>'
            )
        xml_lines.append("</Row>")

    xml_lines.extend(["</Table>", "</Worksheet>", "</Workbook>"])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(xml_lines))


def case_results_to_rows(results: list[CaseResult]) -> tuple[list[str], list[list[str]], list[bool]]:
    headers = [
        "flag",
        "case_id",
        "dims(s,o,u,t)",
        "discrete_evo",
        "ct_evo(drift, diff_coef, diff_cov)",
        "observation_kind",
        "runner",
        "filter_type",
        "discretizer",
        "expected",
        "actual",
        "expected_error",
        "actual_error",
    ]
    rows: list[list[str]] = []
    mismatch_flags: list[bool] = []
    for r in results:
        mismatch = r.expected_pass != r.actual_pass
        mismatch_flags.append(mismatch)
        rows.append(
            [
                "MISMATCH" if mismatch else "ok",
                r.case_id,
                f"({r.model.init_rank},{r.model.observation_rank},{_u_label(r.control_rank)},{r.timesteps})",
                _discrete_evo_for_case(r),
                _ct_evo_for_case(r),
                r.model.observation_kind,
                r.inference.runner,
                r.inference.filter_type or "-",
                r.inference.discretizer,
                "pass" if r.expected_pass else "fail",
                "pass" if r.actual_pass else "fail",
                r.expected_error or "-",
                r.actual_error or "-",
            ]
        )
    return headers, rows, mismatch_flags


def predictive_results_to_rows(
    results: list[PredictiveCaseResult],
) -> tuple[list[str], list[list[str]], list[bool]]:
    headers = [
        "flag",
        "case_id",
        "dims(s,o,u,t)",
        "discrete_evo",
        "ct_evo(drift, diff_coef, diff_cov)",
        "observation_kind",
        "runner",
        "filter_type",
        "discretizer",
        "source_case_id",
        "model_family",
        "initial_kind",
        "context_mode",
        "expected",
        "actual",
        "expected_error",
        "actual_error",
    ]
    rows: list[list[str]] = []
    mismatch_flags: list[bool] = []
    for r in results:
        mismatch = r.expected_pass != r.actual_pass
        mismatch_flags.append(mismatch)
        rows.append(
            [
                "MISMATCH" if mismatch else "ok",
                r.case_id,
                f"({r.init_rank},{r.observation_rank},{_u_label(r.control_rank)},{r.timesteps})",
                _discrete_evo_for_predictive_case(r),
                _ct_evo_for_predictive_case(r),
                r.observation_kind,
                r.runner,
                r.filter_type or "-",
                r.discretizer,
                r.source_case_id,
                r.model_family,
                r.initial_kind,
                _short_context_mode(r.context_mode),
                "pass" if r.expected_pass else "fail",
                "pass" if r.actual_pass else "fail",
                r.expected_error or "-",
                r.actual_error or "-",
            ]
        )
    return headers, rows, mismatch_flags


def _u_label(control_rank: int) -> str:
    if control_rank == 0:
        return "none"
    if control_rank == 1:
        return "1"
    if control_rank == 2:
        return "2"
    return str(control_rank)


def _discrete_evo_for_case(r: CaseResult) -> str:
    if r.model.family == "continuous":
        return "-"
    return r.model.transition_kind


def _ct_evo_for_case(r: CaseResult) -> str:
    if r.model.family != "continuous":
        return "-"
    return f"({r.model.transition_kind},{r.model.diffusion_coeff},{r.model.diffusion_cov})"


def _discrete_evo_for_predictive_case(r: PredictiveCaseResult) -> str:
    if r.model_family == "continuous":
        return "-"
    return r.transition_kind


def _ct_evo_for_predictive_case(r: PredictiveCaseResult) -> str:
    if r.model_family != "continuous":
        return "-"
    return "(zero_ct,eye,eye)"


def _short_context_mode(context_mode: str) -> str:
    mapping = {
        "obs_times": "t",
        "obs_times_obs_values": "t+y",
        "obs_times_obs_values_ctrl_times_ctrl_values": "t+y+u",
    }
    return mapping.get(context_mode, context_mode)


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    expected_idx = headers.index("expected") if "expected" in headers else None
    actual_idx = headers.index("actual") if "actual" in headers else None

    sortable_rows = list(rows)
    # Keep the terminal output focused: hide expected-fail/actual-fail rows.
    if expected_idx is not None and actual_idx is not None:
        sortable_rows = [
            row
            for row in sortable_rows
            if not (
                row[expected_idx].strip().lower() == "fail"
                and row[actual_idx].strip().lower() == "fail"
            )
        ]
    def _priority(row: list[str]) -> tuple[int, str]:
        is_mismatch = row[0] == "MISMATCH"
        expected_is_pass = (
            expected_idx is not None and row[expected_idx].strip().lower() == "pass"
        )
        actual_is_fail = (
            actual_idx is not None and row[actual_idx].strip().lower() == "fail"
        )
        # Most surprising first: expected pass -> actual fail mismatches.
        if is_mismatch and expected_is_pass and actual_is_fail:
            return (0, row[1])
        if is_mismatch:
            return (1, row[1])
        return (2, row[1])

    sortable_rows.sort(key=_priority)

    widths = [
        max(len(headers[idx]), max((len(row[idx]) for row in sortable_rows), default=0))
        for idx in range(len(headers))
    ]
    header_line = " | ".join(
        _paint(headers[idx].ljust(widths[idx]), ANSI_CYAN + ANSI_BOLD)
        for idx in range(len(headers))
    )
    sep_line = "-+-".join("-" * width for width in widths)
    lines = [header_line, sep_line]

    for row in sortable_rows:
        is_mismatch = row[0] == "MISMATCH"
        cells = []
        for idx, cell in enumerate(row):
            padded = cell.ljust(widths[idx])
            if idx == 0:
                padded = _flag_text(is_mismatch, widths[idx])
            elif headers[idx] in {"expected", "actual"}:
                padded = _status_text(cell == "pass", widths[idx])
            elif is_mismatch and headers[idx] in {"expected_error", "actual_error"} and cell != "-":
                padded = _paint(padded, ANSI_YELLOW)
            cells.append(padded)
        lines.append(" | ".join(cells))
    return "\n".join(lines)

