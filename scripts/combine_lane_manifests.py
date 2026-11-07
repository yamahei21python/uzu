#!/usr/bin/env python3
"""Combine sharded lane manifests into one table.

A sweep is split into ``<lane>-shard1of3`` ... so a finished shard keeps its
results. This merges them back, rejects duplicates, and re-checks the invariants
that run_lane enforces per shard.

Usage:
    python scripts/combine_lane_manifests.py --root out --pattern 'robustness-08b-shard*'
    python scripts/combine_lane_manifests.py --root out --pattern 'clean-suite-08b*' --json out/combined.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_manifests(root: Path, pattern: str) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for directory in sorted(root.glob(pattern)):
        manifest = directory / "manifest.json"
        if not manifest.exists():
            print(f"skip {directory}: no manifest.json")
            continue
        found.append((directory, json.loads(manifest.read_text(encoding="utf-8"))))
    return found


def combine(manifests: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: dict[str, str] = {}
    baseline: dict[str, float] = {}
    lanes: list[str] = []

    for directory, manifest in manifests:
        lanes.append(manifest.get("lane", directory.name))
        problems.extend(manifest.get("problems", []))
        for result in manifest.get("results", []):
            name = result.get("name")
            if name in seen:
                problems.append(f"duplicate condition {name}: {seen[name]} and {directory.name}")
                continue
            seen[name] = directory.name
            for lang, ppl in (result.get("ppl") or {}).items():
                chunks = ppl.get("chunks")
                if not chunks:
                    problems.append(f"{name}/{lang}: no eval result")
                    continue
                value = ppl.get("bf16_ppl")
                if value is None:
                    continue
                if lang not in baseline:
                    baseline[lang] = float(value)
                elif abs(baseline[lang] - float(value)) > 0.05:
                    problems.append(
                        f"{name}/{lang}: bf16_ppl drift {float(value):.2f} vs {baseline[lang]:.2f}"
                    )
            results.append(result)

    return {
        "lanes": lanes,
        "results": results,
        "problems": problems,
        "count": len(results),
    }


def format_table(combined: dict[str, Any]) -> str:
    rows = []
    for result in combined["results"]:
        row = {"condition": result.get("name", "?")}
        for lang in ("en", "ja"):
            ppl = (result.get("ppl") or {}).get(lang) or {}
            row[f"{lang}_bf16"] = ppl.get("bf16_ppl")
            row[f"{lang}_ppl"] = ppl.get("candidate_ppl")
            row[f"{lang}_bpc"] = ppl.get("bpc_candidate")
            row[f"{lang}_chunks"] = ppl.get("chunks")
        rows.append(row)

    lines = ["| condition | EN bf16 | EN int4 | EN BPC | JA bf16 | JA int4 | JA BPC | chunks |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        def fmt(value: Any, digits: int = 2) -> str:
            return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "-"

        lines.append(
            f"| {row['condition']} | {fmt(row['en_bf16'])} | {fmt(row['en_ppl'])} | {fmt(row['en_bpc'], 4)} "
            f"| {fmt(row['ja_bf16'])} | {fmt(row['ja_ppl'])} | {fmt(row['ja_bpc'], 4)} "
            f"| {row['en_chunks'] or row['ja_chunks'] or '-'} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pattern", default="*-shard*")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="exit 1 when problems are found")
    args = parser.parse_args()

    manifests = load_manifests(args.root, args.pattern)
    if not manifests:
        raise SystemExit(f"no manifests under {args.root}/{args.pattern}")
    combined = combine(manifests)
    print(json.dumps(combined, indent=2, ensure_ascii=False)[:400])
    print()
    print(format_table(combined))
    print(f"\nconditions: {combined['count']}  lanes: {', '.join(combined['lanes'])}")
    if combined["problems"]:
        print("PROBLEMS")
        for problem in combined["problems"]:
            print(f"  - {problem}")
        if args.strict:
            raise SystemExit(1)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
