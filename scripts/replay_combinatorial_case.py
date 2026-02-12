#!/usr/bin/env python3
"""Replay a single combinatorial case by case_id.

Examples:
  uv run python scripts/replay_combinatorial_case.py --case-id case_0072
  uv run python scripts/replay_combinatorial_case.py --case-id predictive_012
  uv run python scripts/replay_combinatorial_case.py --case-id case_0042 --thin --case-limit 80
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import asdict, dataclass
from pprint import pformat

from tests.combinatorial.assumptions import expected_outcome
from tests.combinatorial.execution import run_forward_case, run_predictive_case
from tests.combinatorial.matrix import (
    build_forward_model_specs,
    build_inference_specs,
    predictive_context_modes,
    sample_cases,
)
from tests.combinatorial.specs import InferenceSpec, ModelSpec


@dataclass(frozen=True)
class ForwardCase:
    case_id: str
    model_spec: ModelSpec
    inf_spec: InferenceSpec
    timesteps: int


@dataclass(frozen=True)
class PredictiveCase:
    case_id: str
    source_case_id: str
    model_spec: ModelSpec
    inf_spec: InferenceSpec
    timesteps: int
    context_mode: str


def _iter_forward_cases(thin: bool, case_limit: int):
    combos = itertools.product(build_forward_model_specs(), build_inference_specs())
    selected = sample_cases(case_limit=case_limit, combos=combos, thin=thin)
    case_idx = 0
    for model_spec, inf_spec in selected:
        for timesteps in (1, 3):
            yield ForwardCase(
                case_id=f"case_{case_idx:04d}",
                model_spec=model_spec,
                inf_spec=inf_spec,
                timesteps=timesteps,
            )
            case_idx += 1


def _iter_predictive_cases(thin: bool, case_limit: int):
    passing_forward_cases: list[ForwardCase] = []
    for case in _iter_forward_cases(thin=thin, case_limit=case_limit):
        expected_pass, _ = expected_outcome(case.model_spec, case.inf_spec)
        actual_pass = True
        try:
            run_forward_case(case.model_spec, case.inf_spec, timesteps=case.timesteps)
        except Exception:
            actual_pass = False
        if expected_pass and actual_pass:
            passing_forward_cases.append(case)

    case_idx = 0
    for forward_case in passing_forward_cases:
        for context_mode in predictive_context_modes():
            yield PredictiveCase(
                case_id=f"predictive_{case_idx:03d}",
                source_case_id=forward_case.case_id,
                model_spec=forward_case.model_spec,
                inf_spec=forward_case.inf_spec,
                timesteps=forward_case.timesteps,
                context_mode=context_mode,
            )
            case_idx += 1


def _find_forward_case(case_id: str, thin: bool, case_limit: int) -> ForwardCase | None:
    for case in _iter_forward_cases(thin=thin, case_limit=case_limit):
        if case.case_id == case_id:
            return case
    return None


def _find_predictive_case(
    case_id: str, thin: bool, case_limit: int
) -> PredictiveCase | None:
    for case in _iter_predictive_cases(thin=thin, case_limit=case_limit):
        if case.case_id == case_id:
            return case
    return None


def _print_forward_case(case: ForwardCase):
    expected_pass, expected_error = expected_outcome(case.model_spec, case.inf_spec)
    print(f"Replaying forward case: {case.case_id}")
    print(f"timesteps={case.timesteps}")
    print(f"expected_pass={expected_pass} expected_error={expected_error or '-'}")
    print("model_spec:")
    print(pformat(asdict(case.model_spec), sort_dicts=True))
    print("inference_spec:")
    print(pformat(asdict(case.inf_spec), sort_dicts=True))
    print("Executing run_forward_case ...")


def _print_predictive_case(case: PredictiveCase):
    expected_pass, expected_error = expected_outcome(case.model_spec, case.inf_spec)
    print(f"Replaying predictive case: {case.case_id}")
    print(f"source_case_id={case.source_case_id}")
    print(f"timesteps={case.timesteps} context_mode={case.context_mode}")
    print(f"expected_pass={expected_pass} expected_error={expected_error or '-'}")
    print("model_spec:")
    print(pformat(asdict(case.model_spec), sort_dicts=True))
    print("inference_spec:")
    print(pformat(asdict(case.inf_spec), sort_dicts=True))
    print("Executing run_predictive_case ...")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a single forward/predictive combinatorial case."
    )
    parser.add_argument(
        "--case-id",
        required=True,
        help="Case id, e.g. case_0072 or predictive_012.",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "forward", "predictive"),
        default="auto",
        help="Case family; auto infers from case-id prefix.",
    )
    parser.add_argument(
        "--thin",
        action="store_true",
        help="Replay against thinned matrix indexing.",
    )
    parser.add_argument(
        "--case-limit",
        type=int,
        default=220,
        help="Case limit when --thin is used (must match test run).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    kind = args.kind
    if kind == "auto":
        if args.case_id.startswith("predictive_"):
            kind = "predictive"
        elif args.case_id.startswith("case_"):
            kind = "forward"
        else:
            raise ValueError(
                "Could not infer --kind from case_id. Use --kind forward|predictive."
            )

    if kind == "forward":
        case = _find_forward_case(args.case_id, thin=args.thin, case_limit=args.case_limit)
        if case is None:
            raise ValueError(
                f"Case {args.case_id!r} not found for thin={args.thin}, "
                f"case_limit={args.case_limit}."
            )
        _print_forward_case(case)
        run_forward_case(case.model_spec, case.inf_spec, timesteps=case.timesteps)
        print("Forward case executed without exception.")
        return

    case = _find_predictive_case(args.case_id, thin=args.thin, case_limit=args.case_limit)
    if case is None:
        raise ValueError(
            f"Case {args.case_id!r} not found for thin={args.thin}, "
            f"case_limit={args.case_limit}."
        )
    _print_predictive_case(case)
    run_predictive_case(
        case.model_spec,
        case.inf_spec,
        case.context_mode,
        timesteps=case.timesteps,
    )
    print("Predictive case executed without exception.")


if __name__ == "__main__":
    main()

