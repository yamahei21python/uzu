#!/usr/bin/env python3
"""Remove duplicate non-empty lines from a corpus, keeping first occurrence."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    kept: list[str] = []
    removed = 0
    for line in lines:
        key = line.strip()
        if key and key in seen:
            removed += 1
            continue
        if key:
            seen.add(key)
        kept.append(line)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"{args.input} -> {args.output}: removed={removed} kept={len(kept)}")


if __name__ == "__main__":
    main()
