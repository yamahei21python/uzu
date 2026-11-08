#!/usr/bin/env python3
"""Run `cli bench` under a load gate and report medians, not single runs.

The same model has measured 49.0 and 21.0 t/s on this machine depending on what
else was running, which is enough error to invert any comparison. This wrapper:

  * refuses to start until the 1-minute load average is below a threshold
  * runs one warmup trial that is discarded
  * runs N trials and reports median / min / max with the load seen at each
  * writes a summary JSON so later comparisons carry their conditions

Usage:
    python scripts/bench_protocol.py --model /tmp/model --trials 5 --out /tmp/bench.json
    python scripts/bench_protocol.py --model /tmp/model --task experiments/bench/task_count_512.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_TASK = Path(__file__).resolve().parents[1] / "experiments" / "bench" / "task_count_512.json"


def load_average() -> float:
    return os.getloadavg()[0]


def wait_for_idle(max_load: float, max_wait_s: int, poll_s: int = 20) -> float | None:
    """Return the load once it is below the threshold, or None on timeout."""
    started = time.time()
    while True:
        current = load_average()
        if current <= max_load:
            return current
        if time.time() - started > max_wait_s:
            return None
        print(f"load {current:.1f} > {max_load}; waiting ({poll_s}s)", flush=True)
        time.sleep(poll_s)


def run_trial(cli: Path, model: Path, task: Path, out: Path) -> dict:
    subprocess.run([str(cli), "bench", str(model), str(task), str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    runs = json.loads(out.read_text())
    return {
        "load": load_average(),
        "generate": [run["generate_tokens_per_second"] for run in runs],
        "prompt": [run["prompt_tokens_per_second"] for run in runs],
        "ttft": [run["time_to_first_token"] for run in runs],
        "memory_bytes": runs[0]["memory_used"],
    }


def summarize(trials: list[dict]) -> dict:
    generate = [value for trial in trials for value in trial["generate"]]
    prompt = [value for trial in trials for value in trial["prompt"]]
    ttft = [value for trial in trials for value in trial["ttft"]]
    return {
        "generate_median": statistics.median(generate),
        "generate_min": min(generate),
        "generate_max": max(generate),
        "prompt_median": statistics.median(prompt),
        "ttft_median": statistics.median(ttft),
        "memory_bytes": trials[0]["memory_bytes"],
        "loads": [round(trial["load"], 2) for trial in trials],
        "trials": len(trials),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    parser.add_argument("--cli", type=Path, default=Path("target/release/cli"))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-load", type=float, default=2.0)
    parser.add_argument("--max-wait", type=int, default=1800, help="seconds to wait for idle")
    parser.add_argument("--out", type=Path, default=Path("/tmp/bench_protocol.json"))
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    load = wait_for_idle(args.max_load, args.max_wait)
    if load is None:
        raise SystemExit(
            f"load stayed above {args.max_load} for {args.max_wait}s; not measuring "
            "(a hot machine has produced 2x errors here)"
        )

    scratch = args.out.with_suffix(".trial.json")
    if not args.skip_warmup:
        print("warmup (discarded)", flush=True)
        run_trial(args.cli, args.model, args.task, scratch)

    trials = []
    for index in range(args.trials):
        trial = run_trial(args.cli, args.model, args.task, scratch)
        trials.append(trial)
        print(f"trial {index + 1}/{args.trials} load={trial['load']:.1f} "
              f"generate={statistics.median(trial['generate']):.1f} t/s", flush=True)

    summary = summarize(trials)
    summary["model"] = str(args.model)
    summary["task"] = str(args.task)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    spread = summary["generate_max"] - summary["generate_min"]
    if spread > 0.1 * summary["generate_median"]:
        print(f"warning: spread {spread:.1f} t/s is over 10% of the median", file=sys.stderr)


if __name__ == "__main__":
    main()
