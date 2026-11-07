#!/usr/bin/env python3
"""Prune old Kaggle output directories so the disk does not fill up.

A full disk killed an Antigravity run (`No space left on device`) because
/private/tmp had 30GB of downloaded kernel outputs. This keeps the newest N
directories per prefix and deletes older ones.

    python scripts/cleanup_local.py --dry-run
    python scripts/cleanup_local.py --keep 2
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path("/private/tmp")
PREFIXES = (
    "emb8_kaggle_out",
    "rtn4b_kaggle_out",
    "lane4b",
    "lane_seed",
    "lane4b_v2",
    "eval4b",
    "smoke_out",
    "ag_",
    "kit_cpu",
    "cpu_matrix",
    "cpu_lane",
    "cpu_qad",
)


def candidates(root: Path, prefixes: tuple[str, ...], keep: int) -> list[Path]:
    selected: list[Path] = []
    for prefix in prefixes:
        matches = sorted(
            (path for path in root.glob(f"{prefix}*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        selected.extend(matches[keep:])
    return selected


def size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--keep", type=int, default=1, help="keep the newest N per prefix")
    parser.add_argument("--min-age-hours", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    now = time.time()
    targets = []
    for path in candidates(args.root, PREFIXES, args.keep):
        age_hours = (now - path.stat().st_mtime) / 3600
        if age_hours < args.min_age_hours:
            continue
        targets.append((path, size(path)))

    freed = sum(item[1] for item in targets)
    for path, bytes_used in targets:
        print(f"{'[dry-run] ' if args.dry_run else ''}{path}  {bytes_used/1024**3:.2f} GB")
        if not args.dry_run:
            shutil.rmtree(path, ignore_errors=True)
    print(f"{len(targets)} dirs, {freed/1024**3:.2f} GB {'would be ' if args.dry_run else ''}freed")


if __name__ == "__main__":
    main()
