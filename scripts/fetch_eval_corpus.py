#!/usr/bin/env python3
"""Dump a Hugging Face text dataset split to a plain UTF-8 corpus file.

Used to freeze the evaluation corpus (WikiText-2, Wikipedia, etc.) so BF16 and
quantized runs see exactly the same text.

Usage:
  HF_HUB_ENABLE_XET=0 python scripts/fetch_eval_corpus.py \
    --dataset wikitext --config wikitext-2-raw-v1 --split test \
    --output experiments/qwen35-0.8b/data/wikitext2_test.txt \
    --max-characters 2000000
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-characters", type=int, default=2_000_000)
    parser.add_argument("--skip-documents", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    parts: list[str] = []
    total = 0
    skipped = 0
    for row in dataset:
        if skipped < args.skip_documents:
            skipped += 1
            continue
        text = row.get(args.text_column)
        if not isinstance(text, str) or not text.strip():
            continue
        parts.append(text)
        total += len(text)
        if total >= args.max_characters:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"{args.output}: {total} characters, {len(parts)} documents")


if __name__ == "__main__":
    main()
