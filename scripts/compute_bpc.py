#!/usr/bin/env python3
"""Normalize evaluation perplexity to bits-per-character (BPC).

PPL is not comparable across languages or tokenizers: a tokenizer with higher
fertility (tokens per character) spreads the same text over more tokens, which
inflates PPL. BPC divides the total negative log-likelihood by characters
instead:

    BPC = log2(PPL) * tokens / chars = log2(PPL) * fertility

Only the public tokenizer is needed. No model weights, no GPU.

The character count is measured on the exact token stream that was scored:
the chunks are rebuilt with the same ``chunks_from_text`` logic as
``eval_quantized.py`` and decoded once as a contiguous stream, so multi-byte
characters split across chunk boundaries are not double counted.

Usage:
    python scripts/compute_bpc.py \
        --ppl-json out/eval/ppl_en.json \
        --ppl-json out/eval/ppl_ja.json \
        --tokenizer Qwen/Qwen3.5-0.8B

    python scripts/compute_bpc.py --ppl-json ppl_en.json --tokenizer ./tokenizer-dir --json report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def load_tokenizer(source: str):
    from tokenizers import Tokenizer

    path = Path(source)
    if path.is_dir():
        return Tokenizer.from_file(str(path / "tokenizer.json"))
    if path.is_file():
        return Tokenizer.from_file(str(path))
    from huggingface_hub import hf_hub_download

    return Tokenizer.from_file(hf_hub_download(source, "tokenizer.json"))


def chunks_from_text(tokenizer, text: str, max_tokens: int, min_tokens: int, limit: int) -> list[list[int]]:
    """Mirror of eval_quantized.chunks_from_text; must stay identical."""
    ids = tokenizer.encode(text, add_special_tokens=True).ids
    chunks = [ids[start : start + max_tokens] for start in range(0, len(ids), max_tokens)]
    chunks = [chunk for chunk in chunks if len(chunk) >= min_tokens]
    return chunks[:limit]


def read_corpus(path: str) -> str:
    text = Path(path).read_text(encoding="utf-8")
    return text.replace("\x00", "")


def char_count(tokenizer, chunks: list[list[int]]) -> int:
    """Decode the scored token stream once and count characters."""
    flat: list[int] = []
    for chunk in chunks:
        flat.extend(chunk)
    return len(tokenizer.decode(flat))


def bpc(ppl: float, tokens: int, chars: int) -> float:
    if chars <= 0:
        raise ValueError("chars must be positive")
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if ppl <= 0:
        raise ValueError("ppl must be positive")
    return math.log2(ppl) * tokens / chars


def sha256_text(path: str) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def analyze(payload: dict[str, Any], tokenizer_source: str, corpus_override: str | None) -> dict[str, Any]:
    config = payload.get("config") or {}
    corpus = corpus_override or config.get("corpus") or payload.get("corpus")
    if not corpus:
        raise SystemExit("no corpus in PPL JSON; pass --corpus")
    max_tokens = int(config.get("max_tokens", 128))
    min_tokens = int(config.get("min_tokens", 16))
    max_sequences = int(config.get("max_sequences", payload.get("chunks", 0) or 0))
    if max_sequences <= 0:
        raise SystemExit("no max_sequences in PPL JSON; pass --max-sequences")

    tokenizer = load_tokenizer(tokenizer_source)
    text = read_corpus(corpus)
    chunks = chunks_from_text(tokenizer, text, max_tokens, min_tokens, max_sequences)
    if not chunks:
        raise SystemExit(f"no chunks from {corpus}")

    tokens = sum(len(chunk) for chunk in chunks)
    chars = char_count(tokenizer, chunks)
    fertility = tokens / chars

    expected_chunks = payload.get("chunks")
    expected_tokens = payload.get("tokens")
    report: dict[str, Any] = {
        "corpus": corpus,
        "corpus_sha256": sha256_text(corpus),
        "tokenizer": tokenizer_source,
        "config": {
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
            "max_sequences": max_sequences,
        },
        "chunks": len(chunks),
        "tokens": tokens,
        "chars": chars,
        "tokens_per_char": fertility,
        "reproduced": {
            "chunks_match": expected_chunks is None or expected_chunks == len(chunks),
            "tokens_match": expected_tokens is None or expected_tokens == tokens,
        },
    }

    for key, value in payload.items():
        if key.endswith("_ppl") and isinstance(value, (int, float)):
            label = key[: -len("_ppl")] or "model"
            report[f"bpc_{label}"] = bpc(float(value), tokens, chars)
            report[f"ppl_{label}"] = float(value)

    if "delta_nll_per_token" in payload:
        report["bpc_delta"] = float(payload["delta_nll_per_token"]) / math.log(2) * fertility

    return report


def format_table(reports: list[dict[str, Any]]) -> str:
    lines = [
        "| run | chunks | tokens | chars | tok/char | PPL | BPC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        name = Path(report.get("source", "?")).parent.name or "?"
        ppl = report.get("ppl_candidate", report.get("ppl_bf16"))
        bp = report.get("bpc_candidate", report.get("bpc_bf16"))
        lines.append(
            f"| {name} | {report['chunks']} | {report['tokens']} | {report['chars']} "
            f"| {report['tokens_per_char']:.4f} | {ppl:.2f} | {bp:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ppl-json", action="append", required=True, help="eval_quantized.py output (repeatable)")
    parser.add_argument("--tokenizer", required=True, help="tokenizer dir, tokenizer.json, or HF repo id")
    parser.add_argument("--corpus", default=None, help="override corpus path recorded in the PPL JSON")
    parser.add_argument("--json", default=None, help="write the report here")
    parser.add_argument("--strict", action="store_true", help="fail when chunk/token counts do not reproduce")
    args = parser.parse_args()

    reports = []
    failed = False
    for path in args.ppl_json:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        report = analyze(payload, args.tokenizer, args.corpus)
        report["source"] = str(path)
        reports.append(report)
        reproduced = report["reproduced"]
        if args.strict and not (reproduced["chunks_match"] and reproduced["tokens_match"]):
            failed = True
            print(f"MISMATCH {path}: {reproduced}")

    for report in reports:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    print()
    print(format_table(reports))

    if args.json:
        Path(args.json).write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
