#!/usr/bin/env python3
"""Reject calibration corpora that overlap with evaluation corpora."""

from __future__ import annotations

import argparse
from pathlib import Path


def long_lines(path: Path, min_chars: int) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(line.strip()) >= min_chars
    }


def overlap_lines(calibration: Path, eval_corpora: list[Path], min_chars: int = 20) -> tuple[int, str | None]:
    blocked: set[str] = set()
    for path in eval_corpora:
        blocked |= long_lines(path, min_chars)
    if not blocked:
        return 0, None
    overlap = long_lines(calibration, min_chars) & blocked
    if not overlap:
        return 0, None
    return len(overlap), next(iter(overlap))[:120]


def assert_disjoint(
    calibration: Path,
    eval_corpora: list[Path],
    *,
    min_chars: int = 20,
    allow_overlap: bool = False,
) -> None:
    if not eval_corpora:
        if allow_overlap:
            return
        raise SystemExit(
            "at least one --eval-corpus is required; use --allow-overlap only for deliberate analysis"
        )
    for path in eval_corpora:
        if not path.exists():
            raise SystemExit(f"missing eval corpus: {path}")
    count, sample = overlap_lines(calibration, eval_corpora, min_chars)
    if count and not allow_overlap:
        raise SystemExit(
            f"calibration corpus overlaps evaluation corpora: {count} lines "
            f"(e.g. {sample!r}). Deduplicate first or pass --allow-overlap."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--eval", nargs="+", type=Path, required=True)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--allow-overlap", action="store_true")
    args = parser.parse_args()
    assert_disjoint(
        args.calibration,
        args.eval,
        min_chars=args.min_chars,
        allow_overlap=args.allow_overlap,
    )
    print("corpus guard: ok")


if __name__ == "__main__":
    main()
