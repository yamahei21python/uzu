#!/usr/bin/env python3
"""Pull a results repository and summarise what landed.

The Kaggle operator uploads ``<lane>/manifest.json`` + per-condition JSON to
``<user>/uzu-results``. This downloads it and prints what is complete, what is
missing, and whether any manifest reported problems, so nobody has to eyeball a
directory tree.

Usage:
    python scripts/pull_results.py --repo kentasatomuse/uzu-results --out results/kentasatomuse
    python scripts/pull_results.py --out results/kentasatomuse --skip-download
    python scripts/pull_results.py --repo ... --out ... --json results/summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def load_manifests(root: Path) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(root.rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"skip unreadable {path}")
            continue
        payload["_path"] = str(path)
        manifests.append(payload)
    return manifests


def summarise(root: Path) -> dict[str, Any]:
    manifests = load_manifests(root)
    lanes = []
    total_conditions = 0
    problems: list[str] = []
    for manifest in manifests:
        results = manifest.get("results", [])
        lane = {
            "lane": manifest.get("lane") or Path(manifest["_path"]).parent.name,
            "conditions": len(results),
            "complete": sum(1 for result in results if result.get("status") == "complete"),
            "problems": manifest.get("problems", []),
            "spec": manifest.get("spec"),
            "updated": manifest.get("updated"),
            "path": manifest["_path"],
        }
        total_conditions += lane["conditions"]
        problems.extend(f"{lane['lane']}: {problem}" for problem in lane["problems"])
        lanes.append(lane)
    return {
        "lanes": lanes,
        "lane_count": len(lanes),
        "condition_count": total_conditions,
        "problems": problems,
    }


def format_table(summary: dict[str, Any]) -> str:
    lines = ["| lane | conditions | complete | problems | updated |", "|---|---:|---:|---:|---|"]
    for lane in summary["lanes"]:
        lines.append(
            f"| {lane['lane']} | {lane['conditions']} | {lane['complete']} "
            f"| {len(lane['problems'])} | {lane.get('updated') or '-'} |"
        )
    return "\n".join(lines)


def download(repo: str, out: Path, revision: str | None) -> None:
    command = ["hf", "download", repo, "--repo-type", "dataset", "--local-dir", str(out)]
    if revision:
        command += ["--revision", revision]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=None, help="HF dataset repo, e.g. <user>/uzu-results")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="exit 1 when a manifest reports problems")
    args = parser.parse_args()

    if not args.skip_download:
        if not args.repo:
            parser.error("--repo is required unless --skip-download is set")
        download(args.repo, args.out, args.revision)

    if not args.out.exists():
        raise SystemExit(f"nothing at {args.out}")

    summary = summarise(args.out)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:1500])
    print()
    print(format_table(summary))
    print(f"\nlanes: {summary['lane_count']}  conditions: {summary['condition_count']}")
    if summary["problems"]:
        print("PROBLEMS")
        for problem in summary["problems"]:
            print(f"  - {problem}")
        if args.strict:
            raise SystemExit(1)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
