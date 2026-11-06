#!/usr/bin/env python3
"""Summarize quantization experiment results as Markdown tables.

Scans `experiments/**/` for:
  * logits*.json          (compare_logits.py)
  * ppl_*.json            (eval_quantized.py)
  * reconstruction_weighted*.json (verify_rht_int4_model.py)

Usage:
  python scripts/summarize_results.py experiments/qwen35-0.8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()

    print(f"# Results: {args.root}\n")

    print("## Logits (BF16 reference vs candidate)\n")
    print("| experiment | top1 | top5 | KL | cosine |")
    print("|---|---:|---:|---:|---:|")
    for path in sorted(args.root.glob("**/logits*.json")):
        data = load_json(path)
        if not data:
            continue
        if isinstance(data, list):
            positions = sum(row.get("positions", 0) for row in data)
            if not positions:
                continue
            top1 = sum(row.get("top1_matches", 0) for row in data) / positions
            top5 = sum(row.get("top5_overlap", 0) * row.get("positions", 0) for row in data) / positions
            kl = sum(row.get("kl_bf16_int4", 0) * row.get("positions", 0) for row in data) / positions
            cosine = sum(row.get("cosine_logits", 0) * row.get("positions", 0) for row in data) / positions
        elif "top1_agreement" in data:
            top1 = data["top1_agreement"]
            top5 = data["top5_overlap"]
            kl = data["kl_bf16_candidate"]
            cosine = data["cosine_logits"]
        else:
            continue
        name = str(path.relative_to(args.root).with_suffix(""))
        print(f"| {name} | {top1:.3f} | {top5:.3f} | {kl:.4f} | {cosine:.4f} |")

    print("\n## Perplexity\n")
    print("| experiment | corpus | BF16 PPL | candidate PPL | ΔNLL/token |")
    print("|---|---|---:|---:|---:|")
    for path in sorted(args.root.glob("**/ppl_*.json")):
        data = load_json(path)
        if not data or "bf16_ppl" not in data:
            continue
        name = path.parent.name
        corpus = path.stem.removeprefix("ppl_")
        print(
            f"| {name} | {corpus} | {data['bf16_ppl']:.2f} | {data['candidate_ppl']:.2f} | "
            f"{data['delta_nll_per_token']:.4f} |"
        )

    print("\n## Weighted reconstruction (RTN vs YAQA)\n")
    baseline_candidates = sorted(args.root.glob("rtn-int4-g32/reconstruction_weighted*.json"))
    baseline_path = next(
        (path for path in baseline_candidates if "newcal" in path.name),
        baseline_candidates[0] if baseline_candidates else None,
    )
    if baseline_path is None:
        print("_no RTN baseline_")
        return
    baseline = {layer["path"]: layer for layer in (load_json(baseline_path) or {}).get("layers", [])}
    print("| experiment | median weighted ratio vs RTN | improved / total |")
    print("|---|---:|---:|")
    for path in sorted(args.root.glob("**/reconstruction_weighted*.json")):
        if path == baseline_path or path.parent == baseline_path.parent:
            continue
        data = load_json(path)
        if not data:
            continue
        layers = {layer["path"]: layer for layer in data.get("layers", [])}
        ratios = []
        for key, layer in layers.items():
            base = baseline.get(key, {}).get("weighted_relative")
            value = layer.get("weighted_relative")
            if base and value is not None:
                ratios.append(value / base)
        if not ratios:
            continue
        ratios.sort()
        median = ratios[len(ratios) // 2]
        improved = sum(1 for ratio in ratios if ratio < 1.0)
        name = str(path.relative_to(args.root))
        print(f"| {name} | {median:.3f} | {improved}/{len(ratios)} |")


if __name__ == "__main__":
    main()
