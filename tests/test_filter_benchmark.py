"""Check benchmark parity in CI without imposing a wall-clock threshold."""

from pathlib import Path
from runpy import run_path


def test_filter_runner_benchmark_parity():
    script = Path(__file__).parents[1] / "scripts" / "benchmark_filters.py"
    bench = run_path(str(script))
    args = bench["inputs"](4)
    for cases in bench["runner_cases"]().values():
        bench["measure"](cases, args, repeats=1)
