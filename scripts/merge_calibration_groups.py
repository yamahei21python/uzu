#!/usr/bin/env python3
"""Validate and merge layer-grouped YAQA calibration NPZ files."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np

GROUP_RE = re.compile(r"group-(\d+)-(\d+)\.npz$")
LAYER_RE = re.compile(r"/\[(\d+)\]/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("inputs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = []
    for start in range(0, args.layers, args.group_size):
        end = min(start + args.group_size, args.layers)
        expected.append((start, end))

    seen_ranges: dict[tuple[int, int], Path] = {}
    merged: dict[str, np.ndarray] = {}
    covered_layers: set[int] = set()

    for path in args.inputs:
        match = GROUP_RE.search(path.name)
        if not match:
            raise SystemExit(f"unexpected group filename: {path}")
        start, end = int(match.group(1)), int(match.group(2))
        if (start, end) in seen_ranges:
            raise SystemExit(f"duplicate group {start}-{end}: {seen_ranges[(start, end)]} and {path}")
        seen_ranges[(start, end)] = path

        with np.load(path) as data:
            keys = [key for key in data.files if key != "__meta__"]
            if not keys:
                raise SystemExit(f"empty group: {path}")
            for key in keys:
                layer_match = LAYER_RE.search(key)
                if not layer_match:
                    raise SystemExit(f"key has no layer index: {key} in {path}")
                layer = int(layer_match.group(1))
                if not (start <= layer < end):
                    raise SystemExit(f"key {key} outside group {start}-{end} in {path}")
                if key in merged:
                    raise SystemExit(f"duplicate calibration key: {key}")
                merged[key] = data[key]
                covered_layers.add(layer)

    missing_groups = [f"{start}-{end}" for start, end in expected if (start, end) not in seen_ranges]
    if missing_groups:
        raise SystemExit(f"missing groups: {missing_groups}")
    missing_layers = sorted(set(range(args.layers)) - covered_layers)
    if missing_layers:
        raise SystemExit(f"missing layers: {missing_layers}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp.npz")
    np.savez(tmp, **merged)
    os.replace(tmp, args.output)
    print(
        f"merged {len(seen_ranges)} groups / {len(merged)} arrays / "
        f"{len(covered_layers)} layers -> {args.output}"
    )


if __name__ == "__main__":
    main()
