"""CPU filter benchmarks. Run with `python scripts/benchmark_filters.py`.

For public API comparisons, run the same script with each checkout on PYTHONPATH.
Use --public --save-outputs DIR on the baseline and --public --reference DIR on
the candidate. The candidate checks every selected history before timing.

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
    from dynestyx.inference.custom_filter import (
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
    from dynestyx.inference.custom_filter import run_filter_steps, step_inputs
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


def measure(cases: dict[str, Callable], args, repeats, reverse=False, verify=None):
    names = list(reversed(cases)) if reverse else list(cases)
    compiled, outputs, rows = {}, {}, {}
    for name in names:
        start = perf_counter()
        compiled[name] = jax.jit(cases[name]).lower(*args).compile()
        cold = perf_counter() - start
        outputs[name] = jax.block_until_ready(compiled[name](*args))
        check_equal(outputs[names[0]], outputs[name])
        if verify is not None:
            verify(outputs[name])
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


def public_runner(case):
    """Return the public filter fields consumed by this workload."""
    from typing import cast

    from cd_dynamax.dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered
    from cd_dynamax.src.continuous_discrete_nonlinear_ssm.models import (
        PosteriorCDNLSSMFiltered,
    )
    from cuthbert.ensemble_kalman.ensemble_kalman_filter import EnKFState
    from cuthbert.gaussian.kalman import KalmanFilterState

    from dynestyx.inference.observation_predictions import PredictedObservationOutputs

    def run(a, ys):
        times = jnp.arange(ys.shape[-2], dtype=ys.dtype)
        if case == "continuous":
            dynamics = dsx.LTI_continuous(
                A=-a * jnp.eye(4),
                L=0.2 * jnp.eye(4),
                H=jnp.eye(2, 4),
                R=0.2 * jnp.eye(2),
            )
            config = dsx.configs.ContinuousTimeKFConfig()
        else:
            dynamics = model(a)
            if case == "enkf":
                config = dsx.configs.EnKFConfig(
                    n_particles=32, crn_seed=jax.random.PRNGKey(38)
                )
            else:
                config = dsx.configs.KFConfig(
                    filter_source="cd_dynamax" if case == "cd" else "cuthbert",
                    associative=case == "associative",
                )
        with dsx.Filter(config):
            if case == "plate":
                with dsx.plate("trajectory", ys.shape[0]):
                    result = dsx.condition(
                        "x", dynamics, obs_times=times, obs_values=ys
                    )
            else:
                result = dsx.condition("x", dynamics, obs_times=times, obs_values=ys)
        if case == "grad":
            return result.marginal_loglik
        if case == "enkf":
            states = cast(EnKFState, result.states)
            history = (states.ensemble, states.predicted_ensemble)
        elif case in {"cd", "continuous"}:
            states = cast(
                PosteriorGSSMFiltered | PosteriorCDNLSSMFiltered, result.states
            )
            history = (states.filtered_means, states.filtered_covariances)
        else:
            states = cast(KalmanFilterState, result.states)
            history = (states.mean, states.chol_cov)
        prediction = cast(
            PredictedObservationOutputs | None, result.predicted_observations
        )
        obs_fields = (
            None
            if prediction is None
            else (
                prediction.mean,
                prediction.cov,
                prediction.obs_cov,
                prediction.ensemble,
                prediction.obs_ensemble,
                prediction.noise_cov,
            )
        )
        return result.marginal_loglik, history, obs_fields

    return jax.value_and_grad(run) if case == "grad" else run


def public_bench(opts, report):
    # Each revision runs in its own interpreter. Compare the saved arrays by
    # leaf order and shape before using its timing ratios.
    cases = reversed(opts.public_cases) if opts.reverse else opts.public_cases
    for case in cases:
        lengths = [128] if case in {"enkf", "continuous"} else opts.lengths
        for length in reversed(lengths) if opts.reverse else lengths:
            label = f"{case}/{length}"
            filename = f"{case}-{length}.npz"

            def verify(output):
                arrays = {
                    f"leaf{i}": np.asarray(x)
                    for i, x in enumerate(jax.tree.leaves(output))
                }
                if opts.reference:
                    with np.load(opts.reference / filename) as expected:
                        check_equal(dict(expected), arrays)
                if opts.save_outputs:
                    opts.save_outputs.mkdir(parents=True, exist_ok=True)
                    np.savez(opts.save_outputs / filename, **arrays)

            rows, _ = measure(
                {case: public_runner(case)},
                inputs(length, case == "plate"),
                opts.repeats,
                verify=verify,
            )
            report["cases"][label] = rows
            print(label, rows[case]["median_ms"], flush=True)
            if opts.output:
                opts.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[32, 512, 4096])
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--public",
        action="store_true",
        help="Benchmark dsx.condition on the imported checkout",
    )
    names = ("seq", "associative", "cd", "plate", "grad", "enkf", "continuous")
    parser.add_argument("--public-cases", nargs="+", choices=names, default=names)
    parser.add_argument(
        "--reference", type=Path, help="Validate against baseline .npz histories"
    )
    parser.add_argument(
        "--save-outputs", type=Path, help="Save baseline arrays for parity checks"
    )
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
    if opts.public:
        public_bench(opts, report)
        return
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
