#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger


def _configure_logger(output_dir: Path) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG",
        format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
    )
    log_file = output_dir / "monitor.log"
    logger.add(
        log_file,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {function}:{line} - {message}",
        rotation=None,
        encoding="utf-8",
    )
    logger.debug(f"File logger initialised → {log_file}")


def _read_meminfo() -> dict[str, int]:
    logger.trace("Reading /proc/meminfo")
    values: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as memfile:
        for raw_line in memfile:
            line = raw_line.strip()
            if not line:
                continue
            key, value_part = line.split(":", 1)
            first_token = value_part.strip().split()[0]
            values[key] = int(first_token)
    logger.trace(f"meminfo parsed: {len(values)} keys")
    return values


@dataclass
class CpuStatSnapshot:
    idle: int
    total: int


class CpuPercentSampler:
    def __init__(self) -> None:
        self._last = self._read_cpu_snapshot()
        logger.debug(f"CpuPercentSampler baseline: idle={self._last.idle} total={self._last.total}")

    @staticmethod
    def _read_cpu_snapshot() -> CpuStatSnapshot:
        logger.trace("Reading /proc/stat for CPU snapshot")
        with open("/proc/stat", "r", encoding="utf-8") as statfile:
            line = statfile.readline().strip()
        parts = line.split()
        values = [int(item) for item in parts[1:]]
        idle = values[3] + values[4]
        total = sum(values)
        logger.trace(f"CPU snapshot: idle={idle} total={total}")
        return CpuStatSnapshot(idle=idle, total=total)

    def sample_percent(self) -> float:
        current = self._read_cpu_snapshot()
        delta_idle = current.idle - self._last.idle
        delta_total = current.total - self._last.total
        self._last = current
        if delta_total <= 0:
            logger.warning("CPU delta_total <= 0; returning 0.0% (clock stall?)")
            return 0.0
        busy = delta_total - delta_idle
        percent = max(0.0, min(100.0, 100.0 * busy / delta_total))
        logger.trace(f"CPU sample: {percent:.1f}% (delta_total={delta_total} delta_idle={delta_idle})")
        return percent


def read_ram_used_mb() -> float:
    mem = _read_meminfo()
    total_kb = mem.get("MemTotal", 0)
    avail_kb = mem.get("MemAvailable", 0)
    used_kb = max(0, total_kb - avail_kb)
    used_mb = used_kb / 1024.0
    logger.trace(f"RAM used: {used_mb:.1f} MB (total={total_kb//1024} MB avail={avail_kb//1024} MB)")
    return used_mb


def read_gpu_stats() -> dict[str, float | None]:
    logger.trace("Querying nvidia-smi for GPU stats")
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except FileNotFoundError:
        logger.debug("nvidia-smi not found — GPU stats unavailable")
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None}
    except subprocess.CalledProcessError as exc:
        logger.warning(f"nvidia-smi exited with code {exc.returncode} — GPU stats unavailable")
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None}
    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out — GPU stats unavailable")
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None}

    util_values: list[float] = []
    mem_values: list[float] = []

    for line in completed.stdout.strip().splitlines():
        tokens = [item.strip() for item in line.split(",")]
        if len(tokens) != 2:
            logger.warning(f"Unexpected nvidia-smi output line (skipping): {line!r}")
            continue
        try:
            util_values.append(float(tokens[0]))
            mem_values.append(float(tokens[1]))
        except ValueError:
            logger.warning(f"Could not parse nvidia-smi tokens {tokens!r} — skipping")
            continue

    if not util_values:
        logger.warning("nvidia-smi returned no parseable GPU rows")
        return {"gpu_util_percent": None, "gpu_mem_used_mb": None}

    result = {
        "gpu_util_percent": max(util_values),
        "gpu_mem_used_mb": sum(mem_values),
    }
    logger.trace(f"GPU stats: util={result['gpu_util_percent']:.1f}% mem={result['gpu_mem_used_mb']:.1f} MB ({len(util_values)} device(s))")
    return result


def collect_test_nodeids(target: str, extra_pytest_args: list[str]) -> list[str]:
    cmd = ["pytest", "--collect-only", "-q", target, *extra_pytest_args]
    logger.info(f"Collecting tests: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    nodeids: list[str] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line or "::" not in line:
            continue
        if line.startswith("=") or line.startswith("-"):
            continue
        nodeids.append(line)
    logger.info(f"Collected {len(nodeids)} test node IDs from target={target!r}")
    for i, nid in enumerate(nodeids, 1):
        logger.debug(f"  [{i:>3}] {nid}")
    return nodeids


def run_one_test(
    test_nodeid: str,
    output_dir: Path,
    interval_seconds: float,
    started_at: float,
    cpu_sampler: CpuPercentSampler,
    samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    env: dict[str, str],
) -> tuple[int, float, float]:
    test_name_safe = (
        test_nodeid.replace("/", "_").replace("::", "__").replace("[", "_").replace("]", "_")
    )
    log_path = output_dir / "logs" / f"{test_name_safe}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_t = time.time() - started_at
    transitions.append({"event": "test_start", "t": start_t, "test": test_nodeid})
    logger.info(f"Starting test: {test_nodeid} (t={start_t:.3f}s)")
    logger.debug(f"  log → {log_path}")

    sample_count_before = len(samples)

    with open(log_path, "w", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            ["pytest", "-q", test_nodeid],
            stdout=logfile,
            stderr=subprocess.STDOUT,
            env=env,
        )
        logger.debug(f"  PID={process.pid}")

        poll_n = 0
        while process.poll() is None:
            sample_t = time.time() - started_at
            gpu = read_gpu_stats()
            cpu_pct = cpu_sampler.sample_percent()
            ram_mb = read_ram_used_mb()
            samples.append(
                {
                    "t": sample_t,
                    "test": test_nodeid,
                    "cpu_percent": cpu_pct,
                    "ram_used_mb": ram_mb,
                    "gpu_util_percent": gpu["gpu_util_percent"],
                    "gpu_mem_used_mb": gpu["gpu_mem_used_mb"],
                }
            )
            poll_n += 1
            if poll_n % 10 == 0:
                logger.debug(
                    f"  poll #{poll_n} | t={sample_t:.1f}s | CPU={cpu_pct:.1f}% | RAM={ram_mb:.0f} MB"
                    + (f" | GPU={gpu['gpu_util_percent']:.0f}% {gpu['gpu_mem_used_mb']:.0f} MB" if gpu["gpu_util_percent"] is not None else "")
                )
            time.sleep(interval_seconds)

        exit_code = process.wait()

    end_t = time.time() - started_at
    duration = end_t - start_t
    new_samples = len(samples) - sample_count_before
    transitions.append({"event": "test_end", "t": end_t, "test": test_nodeid, "exit_code": exit_code})

    status = "PASSED" if exit_code == 0 else f"FAILED (exit={exit_code})"
    log_fn = logger.info if exit_code == 0 else logger.error

    test_samples = samples[sample_count_before:]
    if test_samples:
        peak_ram = max(s["ram_used_mb"] for s in test_samples)
        avg_cpu = sum(s["cpu_percent"] for s in test_samples) / len(test_samples)
        gpu_vals = [s["gpu_mem_used_mb"] for s in test_samples if s["gpu_mem_used_mb"] is not None]
        gpu_str = f" | peak GPU VRAM={max(gpu_vals):.0f} MB" if gpu_vals else ""
        resource_str = f"peak RAM={peak_ram:.0f} MB | avg CPU={avg_cpu:.1f}%{gpu_str}"
    else:
        resource_str = "no samples"

    log_fn(f"{status} | {test_nodeid}")
    log_fn(f"  duration={duration:.2f}s | {resource_str} | samples={new_samples}")
    return exit_code, start_t, end_t


def plot_memory_timeline(
    samples: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    logger.info(f"Plotting memory timeline ({len(samples)} samples) → {output_path}")
    if not samples:
        raise RuntimeError("No samples recorded; cannot plot timeline")

    times = [sample["t"] for sample in samples]
    ram = [sample["ram_used_mb"] for sample in samples]
    gpu_mem = [sample["gpu_mem_used_mb"] if sample["gpu_mem_used_mb"] is not None else 0.0 for sample in samples]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, ram, label="RAM used (MB)", linewidth=1.8)
    ax.plot(times, gpu_mem, label="GPU memory used (MB)", linewidth=1.8)

    n_vlines = 0
    for event in transitions:
        if event.get("event") != "test_start":
            continue
        ax.axvline(event["t"], linestyle="--", linewidth=0.7, alpha=0.35, color="black")
        n_vlines += 1

    logger.debug(f"Added {n_vlines} test-start vertical lines to plot")

    ax.set_title(f"Resource timeline across pytest tests{title_suffix}")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Memory (MB)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info(f"Plot saved: {output_path}")


def _log_summary(
    test_results: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    top_n: int = 5,
) -> None:
    if not test_results:
        logger.warning("No test results to summarise")
        return

    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    # --- slowest tests ---
    by_duration = sorted(test_results, key=lambda r: r["duration_s"], reverse=True)
    logger.info(f"Slowest {min(top_n, len(by_duration))} tests (by wall-clock duration):")
    for rank, result in enumerate(by_duration[:top_n], start=1):
        status = "PASS" if result["exit_code"] == 0 else "FAIL"
        logger.info(f"  #{rank:>2}  {result['duration_s']:>8.2f}s  [{status}]  {result['test']}")

    # --- memory heaviest tests (peak RAM during test window) ---
    if samples:
        # build a lookup: test nodeid → peak RAM MB over its sample window
        peak_ram: dict[str, float] = {}
        for sample in samples:
            test = sample["test"]
            ram = sample["ram_used_mb"]
            if ram > peak_ram.get(test, 0.0):
                peak_ram[test] = ram

        by_ram = sorted(peak_ram.items(), key=lambda kv: kv[1], reverse=True)
        logger.info(f"Memory-heaviest {min(top_n, len(by_ram))} tests (peak RAM during test):")
        for rank, (test, peak) in enumerate(by_ram[:top_n], start=1):
            logger.info(f"  #{rank:>2}  {peak:>9.1f} MB  {test}")

        # --- GPU memory heaviest (if GPU data present) ---
        gpu_samples = [s for s in samples if s.get("gpu_mem_used_mb") is not None]
        if gpu_samples:
            peak_gpu: dict[str, float] = {}
            for sample in gpu_samples:
                test = sample["test"]
                gpu_mb = sample["gpu_mem_used_mb"]
                if gpu_mb > peak_gpu.get(test, 0.0):
                    peak_gpu[test] = gpu_mb
            by_gpu = sorted(peak_gpu.items(), key=lambda kv: kv[1], reverse=True)
            logger.info(f"GPU memory-heaviest {min(top_n, len(by_gpu))} tests (peak VRAM during test):")
            for rank, (test, peak) in enumerate(by_gpu[:top_n], start=1):
                logger.info(f"  #{rank:>2}  {peak:>9.1f} MB  {test}")

    logger.info("=" * 60)


def plot_duration_histogram(
    test_results: list[dict[str, Any]],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    logger.info(f"Plotting duration histogram ({len(test_results)} tests) → {output_path}")
    durations = [r["duration_s"] for r in test_results]
    labels = [r["test"].split("::")[-1] for r in test_results]

    fig, ax = plt.subplots(figsize=(max(10, len(durations) * 0.4), 6))
    colors = ["#d9534f" if r["exit_code"] != 0 else "#5b9bd5" for r in test_results]
    bars = ax.bar(range(len(durations)), durations, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Duration (s)")
    ax.set_title(f"Test duration (blue=pass, red=fail){title_suffix}")
    ax.bar_label(bars, fmt="%.1fs", fontsize=7, padding=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info(f"Duration histogram saved: {output_path}")


def plot_memory_histogram(
    test_results: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    output_path: Path,
    title_suffix: str = "",
) -> None:
    logger.info(f"Plotting memory histogram (CPU RAM + GPU VRAM) → {output_path}")

    peak_ram: dict[str, float] = {}
    peak_gpu: dict[str, float] = {}
    for sample in samples:
        test = sample["test"]
        ram = sample["ram_used_mb"]
        if ram > peak_ram.get(test, 0.0):
            peak_ram[test] = ram
        gpu_mb = sample.get("gpu_mem_used_mb")
        if gpu_mb is not None and gpu_mb > peak_gpu.get(test, 0.0):
            peak_gpu[test] = gpu_mb

    if not peak_ram:
        logger.warning("No memory samples available; skipping memory histogram")
        return

    ordered = [r["test"] for r in test_results if r["test"] in peak_ram]
    labels = [t.split("::")[-1] for t in ordered]
    ram_vals = [peak_ram[t] for t in ordered]
    gpu_vals = [peak_gpu.get(t, 0.0) for t in ordered]
    has_gpu = any(v > 0.0 for v in gpu_vals)

    x = range(len(ordered))
    fig, ax = plt.subplots(figsize=(max(10, len(ordered) * 0.4), 6))

    if has_gpu:
        ax.bar(x, ram_vals, label="Peak RAM (MB)", color="#5cb85c")
        ax.bar(x, gpu_vals, bottom=ram_vals, label="Peak GPU VRAM (MB)", color="#f0ad4e")
        ax.legend(loc="upper right")
        logger.debug(f"GPU VRAM data present for {sum(1 for v in gpu_vals if v > 0)} tests")
    else:
        ax.bar(x, ram_vals, color="#5cb85c")
        logger.debug("No GPU VRAM data; showing RAM only")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Peak memory used (MB)")
    ax.set_title(f"Peak memory per test — RAM + GPU VRAM (stacked){title_suffix}")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info(f"Memory histogram saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run pytest tests one by one while collecting CPU/RAM/GPU telemetry, "
            "write JSON output, and save a memory timeline PNG."
        )
    )
    parser.add_argument("--target", default="tests", help="Pytest target to collect tests from (default: tests)")
    parser.add_argument("--interval", type=float, default=0.5, help="Sampling interval in seconds (default: 0.5)")
    parser.add_argument("--max-tests", type=int, default=None, help="Optional cap on number of collected tests")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; default is .output/resource_monitor_<timestamp>",
    )
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Extra argument passed to pytest collect and each test run. Repeat as needed.",
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop execution immediately when a test fails",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f".output/resource_monitor_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_logger(output_dir)

    logger.info("=" * 60)
    logger.info("monitor_test_resources starting up")
    logger.info(f"  target         : {args.target}")
    logger.info(f"  interval       : {args.interval}s")
    logger.info(f"  max_tests      : {args.max_tests}")
    logger.info(f"  stop_on_failure: {args.stop_on_failure}")
    logger.info(f"  output_dir     : {output_dir}")
    logger.info(f"  extra pytest   : {args.pytest_arg}")
    logger.info("=" * 60)

    nodeids = collect_test_nodeids(args.target, args.pytest_arg)
    if args.max_tests is not None:
        original_count = len(nodeids)
        nodeids = nodeids[: args.max_tests]
        logger.info(f"Capped test list: {original_count} → {len(nodeids)} (--max-tests={args.max_tests})")

    if not nodeids:
        logger.error("No tests collected — check --target and --pytest-arg values")
        raise RuntimeError("No tests collected; check --target and --pytest-arg values")

    started_at = time.time()
    cpu_sampler = CpuPercentSampler()
    samples: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    test_results: list[dict[str, Any]] = []
    env = dict(os.environ)

    logger.info(f"Starting test loop: {len(nodeids)} tests")

    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    last_snapshot_t = time.time()
    snapshot_interval = 600  # seconds

    for index, nodeid in enumerate(nodeids, start=1):
        logger.info(f"[{index}/{len(nodeids)}] {nodeid}")
        exit_code, start_t, end_t = run_one_test(
            test_nodeid=nodeid,
            output_dir=output_dir,
            interval_seconds=args.interval,
            started_at=started_at,
            cpu_sampler=cpu_sampler,
            samples=samples,
            transitions=transitions,
            env=env,
        )
        test_results.append(
            {
                "test": nodeid,
                "exit_code": exit_code,
                "start_t": start_t,
                "end_t": end_t,
                "duration_s": max(0.0, end_t - start_t),
            }
        )

        if exit_code != 0 and args.stop_on_failure:
            logger.warning(f"--stop-on-failure: halting after failure in {nodeid}")
            break

        now = time.time()
        if now - last_snapshot_t >= snapshot_interval and samples and test_results:
            elapsed_min = (now - started_at) / 60
            snap_ts = datetime.now().strftime("%H%M%S")
            suffix = f"  [snapshot at {elapsed_min:.1f} min]"
            logger.info(f"Saving intermediate plots (t={elapsed_min:.1f} min) → snapshots/{snap_ts}/")
            snap_dir = snapshots_dir / snap_ts
            snap_dir.mkdir(parents=True, exist_ok=True)
            try:
                plot_memory_timeline(samples=samples, transitions=transitions, output_path=snap_dir / "memory_timeline.png", title_suffix=suffix)
                plot_duration_histogram(test_results=test_results, output_path=snap_dir / "histogram_durations.png", title_suffix=suffix)
                plot_memory_histogram(test_results=test_results, samples=samples, output_path=snap_dir / "histogram_memory.png", title_suffix=suffix)
            except Exception as exc:
                logger.warning(f"Intermediate plot failed: {exc}")
            last_snapshot_t = now

    total_elapsed = time.time() - started_at
    failing = [result for result in test_results if result["exit_code"] != 0]
    logger.info(f"Test loop complete: {len(test_results)} run, {len(failing)} failed, elapsed={total_elapsed:.1f}s")

    json_path = output_dir / "resource_timeline.json"
    png_path = output_dir / "memory_timeline.png"
    duration_hist_path = output_dir / "histogram_durations.png"
    memory_hist_path = output_dir / "histogram_memory.png"

    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "target": args.target,
        "interval_seconds": args.interval,
        "test_count": len(test_results),
        "samples": samples,
        "transitions": transitions,
        "tests": test_results,
    }

    logger.info(f"Writing JSON ({len(samples)} samples, {len(transitions)} transitions) → {json_path}")
    with open(json_path, "w", encoding="utf-8") as outfile:
        json.dump(payload, outfile, indent=2)
    logger.info(f"JSON written: {json_path}")

    final_suffix = f"  [final — {total_elapsed / 60:.1f} min total]"
    plot_memory_timeline(samples=samples, transitions=transitions, output_path=png_path, title_suffix=final_suffix)
    plot_duration_histogram(test_results=test_results, output_path=duration_hist_path, title_suffix=final_suffix)
    plot_memory_histogram(test_results=test_results, samples=samples, output_path=memory_hist_path, title_suffix=final_suffix)

    _log_summary(test_results=test_results, samples=samples)

    log_path = output_dir / "monitor.log"
    logger.info(f"Log file         : {log_path}")
    logger.info(f"JSON             : {json_path}")
    logger.info(f"Timeline PNG     : {png_path}")
    logger.info(f"Duration hist    : {duration_hist_path}")
    logger.info(f"Memory hist      : {memory_hist_path}")

    if failing:
        logger.error(f"{len(failing)} test(s) failed:")
        for result in failing:
            logger.error(f"  {result['test']} (exit={result['exit_code']} duration={result['duration_s']:.2f}s)")
        return 1

    logger.success(f"All {len(test_results)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
