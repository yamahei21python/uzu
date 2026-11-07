#!/usr/bin/env python3
"""Generate sharded lane specs for a calibration sweep.

A 27-condition sweep (samples x sequence length x damping) does not fit in one
Kaggle session, and a single spec cannot be resumed per shard. This writes N
specs, each with its own lane name, so a shard that finishes stays finished.

Usage:
    python scripts/make_sweep_spec.py \
        --base lanes/clean-suite-08b.json \
        --out-dir lanes \
        --name robustness-08b \
        --corpus data/calibration_mixed_en_ja_dedup.txt \
        --samples 16 32 64 \
        --seq 128 256 512 \
        --damping 1e-4 1e-3 1e-2 \
        --shards 3
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def condition_name(samples: int, seq: int, damping: float) -> str:
    text = f"{damping:.0e}".replace("e-0", "e-").replace("e+0", "e")
    return f"s{samples}_q{seq}_d{text}"


def build_conditions(samples: list[int], seqs: list[int], dampings: list[float], corpus: str) -> list[dict]:
    conditions = []
    for sample_count in samples:
        for seq in seqs:
            for damping in dampings:
                conditions.append(
                    {
                        "name": condition_name(sample_count, seq, damping),
                        "calibration_corpus": corpus,
                        "samples": sample_count,
                        "max_tokens": seq,
                        "damping": damping,
                    }
                )
    return conditions


def make_specs(base: dict, conditions: list[dict], shards: int, name: str) -> list[dict]:
    specs = []
    for index in range(shards):
        spec = copy.deepcopy(base)
        shard_conditions = [condition for position, condition in enumerate(conditions) if position % shards == index]
        spec["lane"] = f"{name}-shard{index + 1}of{shards}"
        spec["conditions"] = shard_conditions
        spec.pop("merge", None)
        specs.append(spec)
    return specs


def validate(specs: list[dict], conditions: list[dict]) -> list[str]:
    problems = []
    seen: list[str] = []
    for spec in specs:
        for condition in spec["conditions"]:
            if condition["name"] in seen:
                problems.append(f"duplicate condition: {condition['name']}")
            seen.append(condition["name"])
    missing = {condition["name"] for condition in conditions} - set(seen)
    if missing:
        problems.append(f"missing conditions: {sorted(missing)[:5]}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, default=Path("lanes/clean-suite-08b.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("lanes"))
    parser.add_argument("--name", default="robustness-08b")
    parser.add_argument("--corpus", default="data/calibration_mixed_en_ja_dedup.txt")
    parser.add_argument("--samples", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--seq", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--damping", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = json.loads(args.base.read_text(encoding="utf-8"))
    conditions = build_conditions(args.samples, args.seq, args.damping, args.corpus)
    specs = make_specs(base, conditions, args.shards, args.name)

    # per-spec eval max_tokens must follow that spec's conditions
    for spec in specs:
        seqs = {condition["max_tokens"] for condition in spec["conditions"]}
        if len(seqs) == 1:
            spec["eval"]["max_tokens"] = seqs.pop()

    problems = validate(specs, conditions)
    total = sum(len(spec["conditions"]) for spec in specs)
    print(f"conditions: {total} in {len(specs)} shards ({[len(s['conditions']) for s in specs]})")
    if problems:
        print("PROBLEMS")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        path = args.out_dir / f"{spec['lane']}.json"
        print(f"  {path} ({len(spec['conditions'])} conditions)")
        if not args.dry_run:
            path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
