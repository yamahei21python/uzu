#!/usr/bin/env python3
"""Build a reproducible EN/JA mixed calibration corpus.

Doc-level proportional interleave. The output is a single text file so the
existing yaqa_calibration.py / qad_finetune.py `--corpus` path works unchanged.

Example:
  python scripts/make_mixed_corpus.py \
    --en experiments/qwen35-0.8b/data/calibration_en_wikitext2.txt \
    --ja experiments/qwen35-0.8b/data/calibration_ja_wikipedia.txt \
    --ja-ratio 0.65 --target-bytes 480000 --seed 42 \
    --out experiments/qwen35-0.8b/data/calibration_mixed_ja65_en35.txt
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


def read_docs(path: Path) -> list[str]:
    docs = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    docs = [doc for doc in docs if doc]
    if not docs:
        raise SystemExit(f"no non-empty documents in {path}")
    return docs


def take_until(docs: list[str], budget_bytes: int) -> list[str]:
    if budget_bytes <= 0:
        return []
    selected: list[str] = []
    used = 0
    for doc in docs:
        size = len(doc.encode("utf-8")) + 1
        if used + size > budget_bytes and selected:
            break
        selected.append(doc)
        used += size
    return selected


def interleave(ja_docs: list[str], en_docs: list[str]) -> list[str]:
    """Proportional doc-level interleave without long same-language runs."""
    if not ja_docs:
        return list(en_docs)
    if not en_docs:
        return list(ja_docs)
    out: list[str] = []
    i = j = 0
    total_ja = max(len(ja_docs), 1)
    total_en = max(len(en_docs), 1)
    while i < len(ja_docs) or j < len(en_docs):
        take_ja = False
        if j >= len(en_docs):
            take_ja = True
        elif i >= len(ja_docs):
            take_ja = False
        else:
            take_ja = (i + 1) / total_ja <= (j + 1) / total_en
        if take_ja:
            out.append(ja_docs[i])
            i += 1
        else:
            out.append(en_docs[j])
            j += 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", type=Path, required=True)
    parser.add_argument("--ja", type=Path, required=True)
    parser.add_argument("--ja-ratio", type=float, default=0.5)
    parser.add_argument("--target-bytes", type=int, default=480_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-corpus", type=Path, action="append", default=[])
    parser.add_argument("--allow-overlap", action="store_true")
    args = parser.parse_args()

    if not 0.0 <= args.ja_ratio <= 1.0:
        raise SystemExit("--ja-ratio must be within [0, 1]")
    if args.target_bytes <= 0:
        raise SystemExit("--target-bytes must be positive")

    rng = random.Random(args.seed)
    en_docs = read_docs(args.en)
    ja_docs = read_docs(args.ja)
    rng.shuffle(en_docs)
    rng.shuffle(ja_docs)

    ja_target = int(args.target_bytes * args.ja_ratio)
    en_target = args.target_bytes - ja_target
    ja_sel = take_until(ja_docs, ja_target) if ja_target > 0 else []
    en_sel = take_until(en_docs, en_target) if en_target > 0 else []
    if not ja_sel and not en_sel:
        raise SystemExit("ratio/budget produced an empty corpus")

    mixed = interleave(ja_sel, en_sel)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text("\n".join(mixed) + "\n", encoding="utf-8")
    from corpus_guard import assert_disjoint

    assert_disjoint(tmp, args.eval_corpus, allow_overlap=args.allow_overlap)
    os.replace(tmp, args.out)

    ja_bytes = sum(len(doc.encode("utf-8")) for doc in ja_sel)
    en_bytes = sum(len(doc.encode("utf-8")) for doc in en_sel)
    total = ja_bytes + en_bytes
    actual_ratio = ja_bytes / total if total else 0.0
    print(f"docs: ja={len(ja_sel)} en={len(en_sel)}")
    print(f"bytes: ja={ja_bytes} en={en_bytes} total={total}")
    print(f"ja_ratio: requested={args.ja_ratio:.3f} actual={actual_ratio:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
