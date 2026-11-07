#!/usr/bin/env python3
"""Remove exact long-line overlaps between calibration and evaluation corpora."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", nargs="+", type=Path, required=True)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-chars", type=int, default=20)
    return parser.parse_args()


def long_lines(path: Path, min_chars: int) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if len(line.strip()) >= min_chars
    }


def main() -> None:
    args = parse_args()
    blocked: set[str] = set()
    for path in args.eval:
        blocked |= long_lines(path, args.min_chars)

    output_dir = args.output_dir or args.input[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in args.input:
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [line for line in lines if line.strip() not in blocked]
        removed = len(lines) - len(kept)
        output = output_dir / f"{path.stem}_dedup{path.suffix}"
        output.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"{path} -> {output}: removed={removed} kept={len(kept)}")


if __name__ == "__main__":
    main()
