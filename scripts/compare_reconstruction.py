#!/usr/bin/env python3
"""Compare two reconstruction reports from verify_rht_int4_model.py.

Typical use:
  python scripts/compare_reconstruction.py \
    --baseline experiments/.../rtn-int4-g32/reconstruction.json \
    --candidate experiments/.../yaqa-hin-int4-g32/reconstruction.json

Reports per layer:
  * relative weight error change
  * Hessian-weighted objective change
  * how many layers improved
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def load(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text())
    return {layer["path"]: layer for layer in data["layers"]}


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    common = sorted(set(baseline) & set(candidate))
    if not common:
        raise SystemExit("no common layers")

    rows = []
    weight_improved = 0
    weighted_improved = 0
    weighted_total = 0
    for path in common:
        base = baseline[path]
        cand = candidate[path]
        base_weighted = base.get("weighted_relative")
        cand_weighted = cand.get("weighted_relative")

        weight_ratio = None
        if base["relative_rmse"]:
            weight_ratio = cand["relative_rmse"] / base["relative_rmse"]
            if weight_ratio < 1.0:
                weight_improved += 1

        weighted_ratio = None
        if base_weighted is not None and cand_weighted is not None and base_weighted != 0.0:
            weighted_ratio = cand_weighted / base_weighted
            weighted_total += 1
            if weighted_ratio < 1.0:
                weighted_improved += 1

        rows.append(
            {
                "path": path,
                "relative_rmse_baseline": base["relative_rmse"],
                "relative_rmse_candidate": cand["relative_rmse"],
                "relative_rmse_ratio": weight_ratio,
                "weighted_relative_baseline": base_weighted,
                "weighted_relative_candidate": cand_weighted,
                "weighted_relative_ratio": weighted_ratio,
            }
        )

    print(f"layers: {len(common)}")
    print(f"weight error improved: {weight_improved}/{len(common)}")
    if weighted_total:
        print(f"weighted objective improved: {weighted_improved}/{weighted_total}")

    improved = [row for row in rows if row["weighted_relative_ratio"] is not None]
    improved.sort(key=lambda row: row["weighted_relative_ratio"])
    print(f"\nbest {args.top} weighted improvements:")
    for row in improved[: args.top]:
        print(f"  {row['weighted_relative_ratio']:.4f}x  {row['path']}")

    print(f"\nworst {args.top} weighted improvements:")
    for row in improved[-args.top :]:
        print(f"  {row['weighted_relative_ratio']:.4f}x  {row['path']}")

    if args.json_out:
        args.json_out.write_text(json.dumps({"layers": rows}, indent=2) + "\n")
        print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
