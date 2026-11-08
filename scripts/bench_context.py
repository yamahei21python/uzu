#!/usr/bin/env python3
"""Measure prompt (prefill) throughput as the prompt length grows.

Prefill efficiency has two very different failure modes:

  * fixed per-pass overhead (dispatch, norms, RHT) -> t/s rises with length
  * a slow inner loop (the GEMM k-loop)            -> t/s is flat

So the shape of the curve tells us where to look before touching any kernel.
Results are written as JSON so a later run can be compared like for like.

Usage:
    python scripts/bench_context.py --model /tmp/model --lengths 128,512,2048,4096 \
        --out /tmp/context.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Quantization trades numerical precision for memory bandwidth. "
    "A speculative decoder drafts several tokens and verifies them together. "
)


def build_task(tokens: int, generation_tokens: int) -> dict:
    # Roughly 4 characters per token for this filler; the harness tokenizes anyway.
    repeats = max(1, tokens * 4 // len(FILLER))
    return {
        "identifier": f"ctx-{tokens}",
        "repo_id": "local/context",
        "number_of_runs": 1,
        "tokens_limit": generation_tokens,
        "greedy": True,
        "messages": [{"role": "user", "content": FILLER * repeats}],
    }


def measure(cli: Path, model: Path, task_path: Path, out_path: Path) -> dict:
    subprocess.run([str(cli), "bench", str(model), str(task_path), str(out_path)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    runs = json.loads(out_path.read_text())
    run = runs[0]
    prompt_tokens = run["tokens_count_input"]
    prompt_tps = run["prompt_tokens_per_second"]
    # 2 FLOP per parameter per token is the usual approximation; the caller can
    # scale by the model size, so record the raw throughput only.
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_tps": prompt_tps,
        "ttft_s": run["time_to_first_token"],
        "memory_bytes": run["memory_used"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lengths", default="128,512,2048,4096")
    parser.add_argument("--generation-tokens", type=int, default=8)
    parser.add_argument("--params-billions", type=float, default=None,
                        help="optional: compute effective TFLOP/s (2 FLOP per param)")
    parser.add_argument("--cli", type=Path, default=Path("target/release/cli"))
    parser.add_argument("--out", type=Path, default=Path("/tmp/bench_context.json"))
    args = parser.parse_args()

    lengths = [int(value) for value in args.lengths.split(",") if value.strip()]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for length in lengths:
            task_path = Path(tmp) / f"task-{length}.json"
            task_path.write_text(json.dumps(build_task(length, args.generation_tokens)), encoding="utf-8")
            row = measure(args.cli, args.model, task_path, Path(tmp) / "out.json")
            row["requested_tokens"] = length
            row["load"] = round(os.getloadavg()[0], 2)
            if args.params_billions:
                row["tflops"] = round(2 * args.params_billions * 1e9 * row["prompt_tps"] / 1e12, 3)
            rows.append(row)
            extra = f" {row['tflops']:.2f} TFLOP/s" if "tflops" in row else ""
            print(f"{length:>5} req -> {row['prompt_tokens']:>5} tok  "
                  f"{row['prompt_tps']:>7.1f} t/s  load={row['load']}{extra}", flush=True)

    args.out.write_text(json.dumps({"model": str(args.model), "rows": rows}, indent=2) + "\n",
                        encoding="utf-8")

    rates = [row["prompt_tps"] for row in rows]
    if len(rates) > 1 and statistics.median(rates) > 0:
        trend = rates[-1] / rates[0]
        print(f"trend (last/first): {trend:.2f}x")
        if trend > 1.3:
            print("=> throughput rises with length: fixed per-pass overhead is significant")
        elif trend < 1.1:
            print("=> throughput is flat: the inner loop itself is the limit")


if __name__ == "__main__":
    main()
