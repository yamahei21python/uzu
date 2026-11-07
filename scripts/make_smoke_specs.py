#!/usr/bin/env python3
"""Derive tiny smoke specs from the real lane specs.

A full lane takes hours; a broken flag, missing file, or merge mismatch shows up
in the first minutes. This shrinks every lane (samples 1, 32 tokens, 2 eval
chunks) so one CPU session can prove that all of them run end to end.

    python scripts/make_smoke_specs.py --out-dir lanes

The generated files are named ``smoke-<lane>.json``. ``balanced-h`` keeps its
merge but only at ratio 0.5; robustness shards keep only the first two
conditions.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
from pathlib import Path

DEFAULT_SKIP = {"smoke-08b"}


def shrink(spec: dict, robustness_limit: int) -> dict:
    spec = copy.deepcopy(spec)
    spec["lane"] = f"smoke-{spec['lane']}"
    spec.setdefault("paths", {})["jax_platform"] = "cpu"
    spec.setdefault("eval", {})["max_sequences"] = 2
    spec["eval"]["max_tokens"] = 32
    spec["eval"]["min_tokens"] = 4
    spec["eval"]["dtype"] = "float32"
    spec.setdefault("convert", {})["parallel"] = "replicated"
    # quantize only layer 0 so a smoke conversion takes seconds, not ten minutes
    spec["convert"]["layer_filter"] = "/[0]/"

    is_robustness = "robustness" in spec["lane"]
    conditions = spec.get("conditions", [])
    if is_robustness:
        conditions = conditions[:robustness_limit]

    for condition in conditions:
        condition["samples"] = 1
        condition["max_tokens"] = 32
        condition.setdefault("calib_dtype", "float32")
    spec["conditions"] = conditions

    merge = spec.get("merge")
    if merge:
        merge["ratios"] = [0.5]
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lanes-dir", type=Path, default=Path("lanes"))
    parser.add_argument("--out-dir", type=Path, default=Path("lanes"))
    parser.add_argument("--robustness-limit", type=int, default=2)
    parser.add_argument("--include", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(args.lanes_dir.glob("*.json")):
        if path.name.startswith("smoke-") or path.stem in DEFAULT_SKIP:
            continue
        if args.include and path.stem not in args.include:
            continue
        if path.name == "bits-down-out-6.json":
            continue
        try:
            spec = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "conditions" not in spec:
            continue
        smoke = shrink(spec, args.robustness_limit)
        out = args.out_dir / f"{smoke['lane']}.json"
        print(f"{path.name} -> {out.name} ({len(smoke['conditions'])} conditions)")
        if not args.dry_run:
            out.write_text(json.dumps(smoke, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written.append(out.name)
    print(f"{len(written)} smoke specs")


if __name__ == "__main__":
    main()
