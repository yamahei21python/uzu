#!/usr/bin/env python3
"""Audit tokenizer behavior on Japanese vs English corpora.

Analyzes:
  A. Fertility breakdown by character category (Kanji, Hiragana, Katakana, Alnum, Symbol, Mixed)
  B. Vocabulary usage (top 50 tokens, >= 3 char tokens, byte-level fallback)
  C. BPC vs BPB vs Bits-per-token interpretation
  D. Corpus character distribution bias

Usage:
  python3 scripts/tokenizer_audit.py \
      --tokenizer Qwen/Qwen3.5-0.8B \
      --corpus-dir data \
      --output-json research/ja_tokenizer_audit.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_tokenizer(source: str):
    from tokenizers import Tokenizer

    path = Path(source)
    if path.is_dir():
        candidate = path / "tokenizer.json"
        if candidate.exists():
            return Tokenizer.from_file(str(candidate))
    if path.is_file():
        return Tokenizer.from_file(str(path))

    from huggingface_hub import hf_hub_download

    file_path = hf_hub_download(source, "tokenizer.json")
    return Tokenizer.from_file(file_path)


def get_char_type(ch: str) -> str:
    """Classify a single unicode character into 5 core types."""
    cp = ord(ch)
    # Kanji: CJK Unified Ideographs, Extensions, Compatibility
    if (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF
        or 0x2F800 <= cp <= 0x2FA1F
    ):
        return "kanji"
    # Hiragana
    if 0x3040 <= cp <= 0x309F:
        return "hiragana"
    # Katakana
    if 0x30A0 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF:
        return "katakana"
    # Alphanumeric
    if ch.isalnum():
        return "alnum"
    # Punctuation, Whitespace, Symbols
    return "symbol"


def get_gpt2_byte_map():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    b2u = dict(zip(bs, [chr(c) for c in cs]))
    u2b = {v: k for k, v in b2u.items()}
    return u2b


def analyze_fertility(tok, text: str) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=True)
    total_tokens = len(enc.ids)
    total_chars = len(text)
    overall_fertility = total_tokens / total_chars if total_chars > 0 else 0.0

    char_sharing = defaultdict(int)
    for s, e in enc.offsets:
        for c in range(s, e):
            char_sharing[c] += 1

    cat_tokens = defaultdict(int)
    cat_chars_raw = defaultdict(int)
    cat_chars_fractional = defaultdict(float)

    for s, e in enc.offsets:
        slice_str = text[s:e]
        types = set(get_char_type(ch) for ch in slice_str)
        if len(types) == 1:
            cat = list(types)[0]
        elif len(types) > 1:
            cat = "mixed"
        else:
            cat = "symbol"  # fallback for empty
        cat_tokens[cat] += 1
        cat_chars_raw[cat] += len(slice_str)
        cat_chars_fractional[cat] += sum(1.0 / char_sharing[c] for c in range(s, e))

    categories_order = ["kanji", "hiragana", "katakana", "alnum", "symbol", "mixed"]
    breakdown = {}
    sum_contribution = 0.0

    for cat in categories_order:
        tk = cat_tokens[cat]
        cr = cat_chars_raw[cat]
        cf = cat_chars_fractional[cat]
        tok_per_char = tk / cf if cf > 0 else 0.0
        char_per_tok = cf / tk if tk > 0 else 0.0
        contribution = tk / total_chars if total_chars > 0 else 0.0
        sum_contribution += contribution

        breakdown[cat] = {
            "tokens": tk,
            "tokens_ratio": tk / total_tokens if total_tokens > 0 else 0.0,
            "chars_raw": cr,
            "chars_fractional": cf,
            "chars_fractional_ratio": cf / total_chars if total_chars > 0 else 0.0,
            "tokens_per_char": tok_per_char,
            "chars_per_token": char_per_tok,
            "contribution": contribution,
        }

    return {
        "chars_total": total_chars,
        "tokens_total": total_tokens,
        "overall_fertility": overall_fertility,
        "sum_contribution": sum_contribution,
        "difference": sum_contribution - overall_fertility,
        "categories": breakdown,
    }


def analyze_vocabulary(tok, text: str) -> dict[str, Any]:
    enc = tok.encode(text, add_special_tokens=True)
    total_tokens = len(enc.ids)

    # 1. Unique token count & Top 50
    counts = Counter(enc.ids)
    unique_count = len(counts)
    top_50_raw = counts.most_common(50)
    top_50 = []
    for tid, freq in top_50_raw:
        dec = tok.decode([tid])
        tok_str = tok.id_to_token(tid)
        top_50.append({
            "id": tid,
            "token_str": tok_str,
            "decoded_str": dec,
            "count": freq,
            "ratio": freq / total_tokens,
        })

    # 2. Tokens occupying >= 3 chars
    count_ge_3 = 0
    cat_ge_3 = defaultdict(int)
    cat_ge_3_samples = defaultdict(Counter)

    kanji_lengths = Counter()

    for tid, (s, e) in zip(enc.ids, enc.offsets):
        slice_str = text[s:e]
        ch_len = len(slice_str)
        if ch_len >= 3:
            count_ge_3 += 1
            types = set(get_char_type(c) for c in slice_str)
            if len(types) == 1:
                cat = list(types)[0]
            else:
                cat = "mixed"
            cat_ge_3[cat] += 1
            cat_ge_3_samples[cat][slice_str] += 1

        # Check kanji compound length
        if ch_len > 0 and all(get_char_type(c) == "kanji" for c in slice_str):
            kanji_lengths[ch_len] += 1

    ge_3_by_cat = {}
    for cat in ["hiragana", "katakana", "mixed", "alnum", "kanji", "symbol"]:
        c_cnt = cat_ge_3[cat]
        ge_3_by_cat[cat] = {
            "count": c_cnt,
            "ratio_of_ge3": c_cnt / count_ge_3 if count_ge_3 > 0 else 0.0,
            "ratio_of_total_tokens": c_cnt / total_tokens if total_tokens > 0 else 0.0,
            "top_examples": [
                {"text": t, "count": cnt}
                for t, cnt in cat_ge_3_samples[cat].most_common(5)
            ],
        }

    total_kanji_tokens = sum(kanji_lengths.values())
    kanji_compound_stats = {
        f"length_{l}": {
            "count": kanji_lengths[l],
            "ratio": kanji_lengths[l] / total_kanji_tokens if total_kanji_tokens > 0 else 0.0,
        }
        for l in sorted(kanji_lengths.keys())
    }

    # 3. Byte-level fallback analysis
    u2b = get_gpt2_byte_map()
    vocab = tok.get_vocab()
    single_byte_token_ids = {
        tid for token_str, tid in vocab.items() if len(token_str) == 1 and token_str in u2b
    }

    ascii_1byte = 0
    non_ascii_1byte = 0
    non_ascii_byte_counts = Counter()

    for tid in enc.ids:
        tok_str = tok.id_to_token(tid)
        if len(tok_str) == 1 and tok_str in u2b:
            b = u2b[tok_str]
            if b < 128:
                ascii_1byte += 1
            else:
                non_ascii_1byte += 1
                non_ascii_byte_counts[(tid, tok_str, b)] += 1

    total_1byte = ascii_1byte + non_ascii_1byte

    # Multi-token character count (offsets sharing)
    char_tokens_count = Counter()
    for s, e in enc.offsets:
        for c in range(s, e):
            char_tokens_count[c] += 1
    chars_with_multi_tokens = sum(1 for c, cnt in char_tokens_count.items() if cnt > 1)

    top_non_ascii_fallbacks = [
        {"id": tid, "token_str": tok_str, "byte_hex": f"0x{b:02x}", "byte_dec": b, "count": cnt}
        for (tid, tok_str, b), cnt in non_ascii_byte_counts.most_common(10)
    ]

    return {
        "unique_tokens_used": unique_count,
        "vocab_size": tok.get_vocab_size(),
        "vocabulary_coverage_ratio": unique_count / tok.get_vocab_size(),
        "top_50_tokens": top_50,
        "tokens_ge_3_chars": {
            "count": count_ge_3,
            "ratio": count_ge_3 / total_tokens,
            "by_category": ge_3_by_cat,
            "kanji_compound_analysis": kanji_compound_stats,
        },
        "byte_level_fallback": {
            "total_1byte_tokens": total_1byte,
            "total_1byte_ratio": total_1byte / total_tokens,
            "ascii_1byte_tokens": ascii_1byte,
            "non_ascii_1byte_fallback_tokens": non_ascii_1byte,
            "non_ascii_fallback_ratio": non_ascii_1byte / total_tokens,
            "chars_with_multi_tokens": chars_with_multi_tokens,
            "top_non_ascii_bytes": top_non_ascii_fallbacks,
        },
    }


def analyze_bpc_and_bpb(tok, corpus_path: str, ppl_bf16: float, ppl_int4: float) -> dict[str, Any]:
    text = open(corpus_path, encoding="utf-8").read().replace("\x00", "")
    ids = tok.encode(text, add_special_tokens=True).ids
    max_tokens = 128
    min_tokens = 16
    limit = 100
    chunks = [ids[i : i + max_tokens] for i in range(0, len(ids), max_tokens)]
    chunks = [c for c in chunks if len(c) >= min_tokens][:limit]

    tokens = sum(max(len(c) - 1, 0) for c in chunks)

    flat = []
    for c in chunks:
        flat.extend(c)
    full_text = tok.decode(flat)
    skipped_chars = sum(len(tok.decode([c[0]])) for c in chunks if c)
    chars = len(full_text) - skipped_chars

    full_bytes = len(full_text.encode("utf-8"))
    skipped_bytes = sum(len(tok.decode([c[0]]).encode("utf-8")) for c in chunks if c)
    bytes_count = full_bytes - skipped_bytes

    bpt_bf16 = math.log2(ppl_bf16)
    bpt_int4 = math.log2(ppl_int4)

    bpc_bf16 = bpt_bf16 * tokens / chars
    bpc_int4 = bpt_int4 * tokens / chars

    bpb_bf16 = bpt_bf16 * tokens / bytes_count
    bpb_int4 = bpt_int4 * tokens / bytes_count

    return {
        "corpus": corpus_path,
        "chunks": len(chunks),
        "tokens": tokens,
        "chars": chars,
        "bytes": bytes_count,
        "tok_per_char": tokens / chars,
        "tok_per_byte": tokens / bytes_count,
        "bytes_per_char": bytes_count / chars,
        "ppl_bf16": ppl_bf16,
        "ppl_int4": ppl_int4,
        "ppl_relative_change": (ppl_int4 - ppl_bf16) / ppl_bf16,
        "bpt_bf16": bpt_bf16,
        "bpt_int4": bpt_int4,
        "bpt_delta": bpt_int4 - bpt_bf16,
        "bpt_relative_change": (bpt_int4 - bpt_bf16) / bpt_bf16,
        "bpc_bf16": bpc_bf16,
        "bpc_int4": bpc_int4,
        "bpc_delta": bpc_int4 - bpc_bf16,
        "bpc_relative_change": (bpc_int4 - bpc_bf16) / bpc_bf16,
        "bpb_bf16": bpb_bf16,
        "bpb_int4": bpb_int4,
        "bpb_delta": bpb_int4 - bpb_bf16,
        "bpb_relative_change": (bpb_int4 - bpb_bf16) / bpb_bf16,
    }


def analyze_char_distribution(text: str) -> dict[str, Any]:
    counts = Counter(get_char_type(ch) for ch in text)
    total = len(text)
    return {
        "total_chars": total,
        "distribution": {
            cat: {
                "count": counts[cat],
                "ratio": counts[cat] / total if total > 0 else 0.0,
            }
            for cat in ["kanji", "hiragana", "katakana", "alnum", "symbol"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Audit Japanese Tokenizer (Fertility, BPC, Bias)")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B", help="Tokenizer name or path")
    parser.add_argument("--corpus-dir", default="data", help="Directory containing corpus files")
    parser.add_argument("--output-json", default="research/ja_tokenizer_audit.json", help="Output JSON path")
    parser.add_argument("--ppl-ja", default="experiments/qwen35-0.8b/clean-enonly-int4-g32/ppl_ja.json")
    parser.add_argument("--ppl-en", default="experiments/qwen35-0.8b/clean-enonly-int4-g32/ppl_en.json")
    args = parser.parse_args()

    tok = load_tokenizer(args.tokenizer)

    # 1. Inspect all corpora in corpus-dir
    corpus_files = sorted(glob.glob(os.path.join(args.corpus_dir, "*.txt")))
    corpora_summary = {}
    for cpath in corpus_files:
        name = Path(cpath).stem
        txt = open(cpath, encoding="utf-8").read().replace("\x00", "")
        enc = tok.encode(txt, add_special_tokens=True)
        corpora_summary[name] = {
            "path": cpath,
            "chars": len(txt),
            "bytes": len(txt.encode("utf-8")),
            "tokens": len(enc.ids),
            "tokens_per_char": len(enc.ids) / len(txt) if len(txt) > 0 else 0.0,
            "tokens_per_byte": len(enc.ids) / len(txt.encode("utf-8")) if len(txt) > 0 else 0.0,
            "bytes_per_char": len(txt.encode("utf-8")) / len(txt) if len(txt) > 0 else 0.0,
        }

    # 2. Detailed fertility breakdown on JA eval
    ja_eval_path = os.path.join(args.corpus_dir, "eval_ja_wikipedia_dedup.txt")
    if not os.path.exists(ja_eval_path):
        # try finding any ja corpus
        ja_candidates = [f for f in corpus_files if "ja" in f]
        if not ja_candidates:
            sys.exit(f"No Japanese corpus found in {args.corpus_dir}")
        ja_eval_path = ja_candidates[0]

    ja_text = open(ja_eval_path, encoding="utf-8").read().replace("\x00", "")
    fertility_ja = analyze_fertility(tok, ja_text)

    # 3. Vocabulary statistics on JA eval
    vocab_ja = analyze_vocabulary(tok, ja_text)

    # 4. BPC interpretation
    en_eval_path = os.path.join(args.corpus_dir, "eval_en_wikitext2_dedup.txt")
    ppl_ja_data = json.load(open(args.ppl_ja)) if os.path.exists(args.ppl_ja) else {}
    ppl_en_data = json.load(open(args.ppl_en)) if os.path.exists(args.ppl_en) else {}

    ja_bf16_ppl = ppl_ja_data.get("bf16_ppl", 43.42138)
    ja_cand_ppl = ppl_ja_data.get("candidate_ppl", 62.51593)
    en_bf16_ppl = ppl_en_data.get("bf16_ppl", 35.37584)
    en_cand_ppl = ppl_en_data.get("candidate_ppl", 39.39458)

    bpc_ja = analyze_bpc_and_bpb(tok, ja_eval_path, ja_bf16_ppl, ja_cand_ppl)
    bpc_en = (
        analyze_bpc_and_bpb(tok, en_eval_path, en_bf16_ppl, en_cand_ppl)
        if os.path.exists(en_eval_path)
        else {}
    )

    ratios = {}
    if bpc_ja and bpc_en:
        ratios = {
            "ppl_cand_ratio_ja_over_en": bpc_ja["ppl_int4"] / bpc_en["ppl_int4"],
            "bpt_cand_ratio_ja_over_en": bpc_ja["bpt_int4"] / bpc_en["bpt_int4"],
            "bpc_cand_ratio_ja_over_en": bpc_ja["bpc_int4"] / bpc_en["bpc_int4"],
            "bpb_cand_ratio_ja_over_en": bpc_ja["bpb_int4"] / bpc_en["bpb_int4"],
            "tokens_per_char_ratio_ja_over_en": bpc_ja["tok_per_char"] / bpc_en["tok_per_char"],
            "tokens_per_byte_ratio_ja_over_en": bpc_ja["tok_per_byte"] / bpc_en["tok_per_byte"],
        }

    # 5. Character composition bias
    en_text = (
        open(en_eval_path, encoding="utf-8").read().replace("\x00", "")
        if os.path.exists(en_eval_path)
        else ""
    )
    dist_ja = analyze_char_distribution(ja_text)
    dist_en = analyze_char_distribution(en_text) if en_text else {}

    report = {
        "tokenizer": args.tokenizer,
        "corpora_summary": corpora_summary,
        "fertility_breakdown_ja": fertility_ja,
        "vocabulary_stats_ja": vocab_ja,
        "bpc_interpretation": {
            "ja": bpc_ja,
            "en": bpc_en,
            "ratios_ja_over_en": ratios,
        },
        "character_distribution_bias": {
            "eval_ja": dist_ja,
            "eval_en": dist_en,
        },
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Audit results written to {args.output_json}")


if __name__ == "__main__":
    main()
