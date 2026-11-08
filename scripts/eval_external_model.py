#!/usr/bin/env python3
"""Evaluate an already-quantized model directory against a BF16 reference.

`run_lane.py` always converts from BF16 first, so it cannot measure a model
that arrives ready-made (an official release, a QAD output, a downloaded
checkpoint). This is the missing half: point it at two directories and it
produces the same ΔBPC JSON that the lanes report.

Both directories must be lalamo-loadable after `make_lalamo_eval_copy.py`; the
script builds the eval copies itself.

Usage:
    python scripts/eval_external_model.py \
        --bf16-dir /kaggle/working/bf16-lalamo \
        --candidate-dir /kaggle/working/official \
        --corpus data/eval_en_wikitext2_dedup.txt --corpus data/eval_ja_wikipedia_dedup.txt \
        --out /kaggle/working/official-eval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LANGS = {"wikitext2": "en", "wikipedia": "ja"}


def language_of(corpus: Path) -> str:
    name = corpus.name.lower()
    for marker, lang in LANGS.items():
        if marker in name:
            return lang
    return name.split("_")[1] if name.startswith("eval_") and "_" in name else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-dir", type=Path, required=True, help="reference BF16 model directory")
    parser.add_argument("--candidate-dir", type=Path, required=True, help="model to measure")
    parser.add_argument("--corpus", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    copies = {}
    for label, source in (("bf16", args.bf16_dir), ("candidate", args.candidate_dir)):
        destination = args.out / f"{label}-eval"
        copies[label] = destination
        command = [
            sys.executable,
            str(HERE / "make_lalamo_eval_copy.py"),
            "--source",
            str(source),
            "--dest",
            str(destination),
        ]
        if args.dry_run:
            print("dry-run:", " ".join(command))
        else:
            subprocess.run(command, check=True)

    results = {}
    for corpus in args.corpus:
        lang = language_of(corpus)
        payload_path = args.out / f"{lang}.json"
        command = [
            sys.executable,
            str(HERE / "eval_quantized.py"),
            "--bf16-dir",
            str(copies["bf16"]),
            "--candidate-dir",
            str(copies["candidate"]),
            "--corpus",
            str(corpus),
            "--max-sequences",
            str(args.max_sequences),
            "--max-tokens",
            str(args.max_tokens),
            "--min-tokens",
            str(args.min_tokens),
            "--dtype",
            args.dtype,
            "--json-out",
            str(payload_path),
        ]
        if args.dry_run:
            print("dry-run:", " ".join(command))
            continue
        subprocess.run(command, check=True)
        payload = json.loads(payload_path.read_text())
        delta = (payload["bpc_candidate"] / payload["bpc_bf16"] - 1) * 100
        results[lang] = {
            "ppl": payload["candidate_ppl"],
            "bf16_ppl": payload["bf16_ppl"],
            "bpc_delta_pct": round(delta, 2),
        }
        print(f"{lang}: {payload['candidate_ppl']:.3f} bpc_delta={delta:+.2f}%", flush=True)

    if args.dry_run:
        return
    (args.out / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("=== FINAL ===", flush=True)
    print(json.dumps(results, indent=1), flush=True)


if __name__ == "__main__":
    main()
