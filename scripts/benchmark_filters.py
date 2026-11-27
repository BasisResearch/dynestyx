"""CPU filter benchmarks. Run with `python scripts/benchmark_filters.py`.

Use --reverse for a fresh trial in the opposite order. Timings include dispatch
and synchronization. Inputs (including the transition coefficient) are dynamic.
"""

import argparse
import json
import platform
import subprocess
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from cuthbert import filter as native_filter

import dynestyx as dsx


def model(a):
    return dsx.LTI_discrete(
        A=a * jnp.eye(4), Q=0.1 * jnp.eye(4), H=jnp.eye(2, 4), R=0.2 * jnp.eye(2)
    )


def sequential_runner(method):
    from dynestyx.inference.filter_interface import (
        FilterOutputs,
        run_filter_steps,
        step_inputs,
    )

    class SplitKF(dsx.configs.SplitFilterConfig):
        def initialize(self, model, inputs, *, key):
            return jnp.zeros(4), jnp.eye(4), jnp.zeros(())

        def predict(self, model, state, inputs, *, key):
            mean, cov, score = state
            transition = model.state_evolution.A
            return (
                transition @ mean,
                transition @ cov @ transition.T + 0.1 * jnp.eye(4),
                score,
            )

        def update(self, model, prior, inputs, *, key):
            mean, cov, score = prior
            h = model.observation_model.H
            residual = inputs.y - h @ mean
            innovation_cov = h @ cov @ h.T + 0.2 * jnp.eye(2)
            gain = jnp.linalg.solve(innovation_cov, h @ cov).T
            increment = jax.scipy.stats.multivariate_normal.logpdf(
                residual, jnp.zeros(2), innovation_cov
            )
            return mean + gain @ residual, cov - gain @ h @ cov, score + increment

        def gather_outputs(self, prior, filtered):
            mean, cov, score = filtered
            return FilterOutputs(score, mean, cov)

    split = SplitKF()

    class CombinedKF(dsx.configs.CombineFilterConfig):
        def init_prepare(self, model, *, key):
            return split.initialize(model, None, key=key)

        def filter_prepare(self, model, inputs, *, key):
            return inputs

        def filter_combine(self, model, state, x):
            prior = jax.lax.cond(
                x.is_first_step,
                lambda: state,
                lambda: split.predict(model, state, x, key=None),
            )
            return split.update(model, prior, x, key=None)

        def gather_outputs(self, filtered):
            return split.gather_outputs(None, filtered)

    combined = CombinedKF()
    config = split if method == "split" else combined

    def run(a, ys):
        dynamics = model(a)
        inputs = step_inputs(
            dynamics, jnp.arange(len(ys), dtype=ys.dtype), ys, None, None
        )
        if method == "manual":

            def body(state, x):
                updated = combined.filter_combine(dynamics, state, x)
                return updated, combined.gather_outputs(updated)

            return jax.lax.scan(
                body, combined.init_prepare(dynamics, key=None), inputs
            )[1]
        return run_filter_steps(
            config.build_filter(dynamics), inputs, None, config.gather_outputs
        )

    return run


def associative_runner(native):
    from dynestyx.inference.filter_interface import run_filter_steps, step_inputs
    from dynestyx.inference.integrations.cuthbert.discrete_filter import (
        build_cuthbert_filter,
    )

    def run(a, ys):
        dynamics = model(a)
        config = dsx.configs.KFConfig(filter_source="cuthbert", associative=True)
        algorithm, _ = build_cuthbert_filter(dynamics, config, None, want_parallel=True)
        inputs = step_inputs(
            dynamics, jnp.arange(len(ys), dtype=ys.dtype), ys, None, None
        )

        def gather(s):
            return s.mean, s.chol_cov, s.log_normalizing_constant

        if native:
            states = native_filter(
                algorithm, inputs, algorithm.init_prepare(), parallel=True
            )
            return jax.tree.map(lambda xs: xs[1:], gather(states))
        return run_filter_steps(algorithm, inputs, None, gather, parallel=True)

    return run


def check_equal(reference, actual):
    if jax.tree.structure(reference) != jax.tree.structure(actual):
        raise AssertionError("Benchmark outputs have different structures")
    for x, y in zip(jax.tree.leaves(reference), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(x, y, rtol=2e-4, atol=2e-4)


def measure(cases: dict[str, Callable], args, repeats, reverse=False):
    names = list(reversed(cases)) if reverse else list(cases)
    compiled, outputs, rows = {}, {}, {}
    for name in names:
        start = perf_counter()
        compiled[name] = jax.jit(cases[name]).lower(*args).compile()
        cold = perf_counter() - start
        outputs[name] = jax.block_until_ready(compiled[name](*args))
        check_equal(outputs[names[0]], outputs[name])
        rows[name] = dict(compile_s=cold, warm_ms=[])

    for _ in range(repeats):
        for name in names:
            start = perf_counter()
            jax.block_until_ready(compiled[name](*args))
            rows[name]["warm_ms"].append(1000 * (perf_counter() - start))
    for row in rows.values():
        samples = row["warm_ms"]
        row.update(
            median_ms=float(np.median(samples)),
            iqr_ms=float(np.percentile(samples, 75) - np.percentile(samples, 25)),
        )
    return rows, outputs


def inputs(length, plated=False):
    shape = (8, length, 2) if plated else (length, 2)
    ys = np.random.default_rng(398).normal(size=shape).astype(np.float32)
    return jax.block_until_ready(jax.device_put((np.float32(0.91), ys)))


def runner_cases():
    return {
        "sequential": {
            name: sequential_runner(name) for name in ("manual", "split", "combined")
        },
        "associative": {
            "native": associative_runner(True),
            "combined": associative_runner(False),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 512, 4096])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path)
    opts = parser.parse_args()
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_compilation_cache", False)
    root = Path(dsx.__file__).resolve().parents[1]
    sha = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    metadata = dict(
        source=str(root),
        sha=sha,
        device=str(jax.devices()[0]),
        host=platform.platform(),
        versions={p: version(p) for p in ("jax", "jaxlib", "cuthbert", "cd-dynamax")},
    )
    report = {"metadata": metadata, "cases": {}}
    for length in opts.lengths:
        for label, cases in runner_cases().items():
            rows, _ = measure(cases, inputs(length), opts.repeats, opts.reverse)
            report["cases"][f"{label}/{length}"] = rows
            print(
                label,
                length,
                {name: round(r["median_ms"], 4) for name, r in rows.items()},
                flush=True,
            )
    if opts.output:
        opts.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
