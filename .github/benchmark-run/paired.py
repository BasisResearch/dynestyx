"""Interleave timings of two checkouts with stdin/stdout JAX workers.

Each worker keeps its compiled function, inputs and device resident outputs.
IPC, parity checks, compilation and JSON encoding are outside the timers.
"""

from pathlib import Path
import json
import sys
import subprocess
import os
from time import perf_counter

# Put the launcher in the benchmark output directory.
root = Path(__file__).resolve().parent
script = Path(os.environ["BENCHMARK"]).resolve()
reference = Path(os.environ["REFERENCE"]).resolve()
baseline = Path(os.environ["BASELINE"]).resolve()


def worker():
    from runpy import run_path

    bench = run_path(str(script))
    jax, np = bench["jax"], bench["np"]
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_compilation_cache", False)
    for line in sys.stdin:
        msg = json.loads(line)
        if msg[0] == "load":
            case, length = msg[1].split("/")
            args = bench["inputs"](int(length), case == "plate")
            f = jax.jit(bench["public_runner"](case))
            t = perf_counter()
            fn = f.lower(*args).compile()
            val = perf_counter() - t
            output = jax.block_until_ready(fn(*args))
            arrays = {
                f"leaf{i}": np.asarray(x) for i, x in enumerate(jax.tree.leaves(output))
            }
            with np.load(reference / f"{case}-{length}.npz") as ref:
                bench["check_equal"](dict(ref), arrays)
        else:
            t = perf_counter()
            jax.block_until_ready(fn(*args))
            val = (perf_counter() - t) * 1000
        print(json.dumps(val), flush=True)


def request(proc, data):
    proc.stdin.write(json.dumps(data) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def med(xs):
    import statistics

    return statistics.median(xs)


def trial(number, order, candidate, cases):
    import statistics

    procs = {}
    rows = {n: {} for n in order}
    try:
        for n in order:
            path = candidate if n == "cand" else baseline
            procs[n] = subprocess.Popen(
                [sys.executable, __file__, "--worker"],
                text=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env={**os.environ, "PYTHONPATH": str(path)},
            )
        for label in cases:
            case, _ = label.split("/")
            for n in order:
                rows[n][label] = {
                    case: {
                        "compile_s": request(procs[n], ["load", label]),
                        "warm_ms": [],
                    }
                }
            for rep in range(31):
                for n in order if rep % 2 == 0 else reversed(order):
                    rows[n][label][case]["warm_ms"].append(request(procs[n], ["run"]))
            for n in order:
                r = rows[n][label][case]
                q1, _, q3 = statistics.quantiles(r["warm_ms"], n=4, method="inclusive")
                r.update(median_ms=med(r["warm_ms"]), iqr_ms=q3 - q1)
                (root / f"{n}{number}.json").write_text(
                    json.dumps({"cases": rows[n]}, indent=2)
                )
            print(
                number,
                label,
                round(
                    rows["cand"][label][case]["median_ms"]
                    / rows["base"][label][case]["median_ms"],
                    3,
                ),
                flush=True,
            )
    finally:
        for proc in procs.values():
            proc.stdin.close()
            proc.wait()


if __name__ == "__main__":
    if sys.argv[1] == "--worker":
        worker()
    else:
        cases = sys.argv[2:]
        trial(2, ["cand", "base"], Path(sys.argv[1]), cases)
        trial(3, ["base", "cand"], Path(sys.argv[1]), list(reversed(cases)))
