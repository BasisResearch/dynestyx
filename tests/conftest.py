from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config._dynestyx_scorecard = []


def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> None:
    if call.when != "call":
        return
    case = getattr(item, "_dynestyx_case", None)
    if case is None:
        return
    actual = "pass" if call.excinfo is None else "fail"
    item.config._dynestyx_scorecard.append(
        {
            "id": case.case_id,
            "expected": case.expected,
            "actual": actual,
            "expected_error": case.expected_error or "None",
        }
    )


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    rows = getattr(config, "_dynestyx_scorecard", [])
    if not rows:
        return
    terminalreporter.write_line("")
    terminalreporter.write_line("dynestyx combinatorial scorecard")
    for row in sorted(rows, key=lambda r: r["id"]):
        terminalreporter.write_line(
            f"{row['id']} | expected={row['expected']} | "
            f"actual={row['actual']} | expected_error={row['expected_error']}"
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("dynestyx combinatorial")
    group.addoption(
        "--dsx-thin-combinatorial",
        action="store_true",
        default=False,
        help="Enable deterministic thinning for combinatorial matrix tests.",
    )
    group.addoption(
        "--dsx-case-limit",
        action="store",
        type=int,
        default=200,
        help="Case limit used when thinning combinatorial matrix tests.",
    )
