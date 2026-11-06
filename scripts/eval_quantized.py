#!/usr/bin/env python3
"""Teacher-forced evaluation: BF16 vs quantized model on a fixed corpus.

Computes, over fixed token chunks:
  * Perplexity (independent chunks; identical chunking for both models)
  * NLL delta per token
  * Top-1 / Top-5 agreement
  * KL(BF16 || candidate)
  * Cosine similarity of logits
  * Top-1 logit margin change

No sampling, no generation: the same token sequence is fed to both models.

Usage:
  JAX_PLATFORM_NAME=cpu python scripts/eval_quantized.py \
    --bf16-dir /tmp/qwen35_0.8b_bf16_lalamo_lalamo_origcfg \
    --candidate-dir /tmp/qwen35_0.8b_rht_int4_lalamo_lalamo_origcfg \
    --corpus experiments/qwen35-0.8b/data/wikitext2_sample.txt \
    --max-sequences 100 --max-tokens 128 \
    --json-out experiments/qwen35-0.8b/rtn-int4-g32/eval_wikitext2.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True, help="UTF-8 text file")
    parser.add_argument("--lalamo-path", default=os.environ.get("LALAMO_PATH", "/tmp/lalamo-direct"))
    parser.add_argument("--max-sequences", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--min-tokens", type=int, default=16)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def chunks_from_text(tokenizer, text: str, max_tokens: int, min_tokens: int, limit: int) -> list[list[int]]:
    ids = tokenizer.encode(text, add_special_tokens=True).ids
    chunks = [ids[start : start + max_tokens] for start in range(0, len(ids), max_tokens)]
    chunks = [chunk for chunk in chunks if len(chunk) >= min_tokens]
    return chunks[:limit]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.lalamo_path)

    import jax
    import jax.numpy as jnp
    import numpy as np

    from lalamo.model import Model
    from lalamo.module import Keychain
    from lalamo.modules import DecoderForwardPassConfig, ForwardPassMode
    from lalamo.utils.sharding import ShardingConfig
    from tokenizers import Tokenizer

    sharding = ShardingConfig.replicated()
    keychain = Keychain.init(seed=0, sharding_config=sharding)
    forward_pass_config = DecoderForwardPassConfig.for_inference(mode=ForwardPassMode.MULTI_TOKEN)

    print("loading BF16 model...")
    bf16 = Model.load(args.bf16_dir, sharding, dtype=jnp.bfloat16)
    print("loading candidate model...")
    candidate = Model.load(args.candidate_dir, sharding, dtype=jnp.bfloat16)
    tokenizer = Tokenizer.from_file(str(args.bf16_dir / "tokenizer.json"))
    print("loaded")

    text = args.corpus.read_text(encoding="utf-8")
    chunks = chunks_from_text(tokenizer, text, args.max_tokens, args.min_tokens, args.max_sequences)
    if not chunks:
        raise SystemExit("no chunks: corpus too small or min-tokens too high")
    print(f"chunks: {len(chunks)}  mean tokens: {sum(len(c) for c in chunks) / len(chunks):.1f}")

    def logits_for(model, ids: list[int]) -> np.ndarray:
        token_ids = jnp.asarray([ids], dtype=jnp.int32)
        positions = jnp.arange(len(ids), dtype=jnp.int32)[None, :]
        result = model.decoder(
            token_ids=token_ids,
            token_positions=positions,
            state=None,
            return_updated_state=False,
            lengths_without_padding=None,
            forward_pass_config=forward_pass_config,
            keychain=keychain.split(1)[0],
        )
        return np.asarray(result.logits[0], dtype=np.float32)

    total_nll_bf16 = 0.0
    total_nll_candidate = 0.0
    total_positions = 0
    top1_matches = 0
    top5_overlap_sum = 0.0
    kl_sum = 0.0
    cosine_sum = 0.0
    margin_delta_sum = 0.0

    for index, ids in enumerate(chunks):
        bf_logits = logits_for(bf16, ids)
        qa_logits = logits_for(candidate, ids)
        # Predict token i+1 from position i.
        bf_pred = bf_logits[:-1]
        qa_pred = qa_logits[:-1]
        targets = np.asarray(ids[1:], dtype=np.int64)

        bf_log_prob = bf_pred - bf_pred.max(axis=-1, keepdims=True)
        bf_log_prob = bf_log_prob - np.log(np.exp(bf_log_prob).sum(axis=-1, keepdims=True))
        qa_log_prob = qa_pred - qa_pred.max(axis=-1, keepdims=True)
        qa_log_prob = qa_log_prob - np.log(np.exp(qa_log_prob).sum(axis=-1, keepdims=True))

        bf_target_log_prob = bf_log_prob[np.arange(len(targets)), targets]
        qa_target_log_prob = qa_log_prob[np.arange(len(targets)), targets]
        total_nll_bf16 -= float(bf_target_log_prob.sum())
        total_nll_candidate -= float(qa_target_log_prob.sum())
        total_positions += len(targets)

        bf_top1 = bf_pred.argmax(axis=-1)
        qa_top1 = qa_pred.argmax(axis=-1)
        top1_matches += int((bf_top1 == qa_top1).sum())

        bf_sorted = np.argsort(-bf_pred, axis=-1)[:, :5]
        qa_sorted = np.argsort(-qa_pred, axis=-1)[:, :5]
        for bf_row, qa_row in zip(bf_sorted, qa_sorted):
            top5_overlap_sum += len(set(bf_row.tolist()) & set(qa_row.tolist())) / 5.0

        bf_prob = np.exp(bf_log_prob)
        qa_prob = np.exp(qa_log_prob)
        kl_sum += float(
            np.sum(bf_prob * (np.log(bf_prob + 1e-45) - np.log(qa_prob + 1e-45)), axis=-1).sum()
        )
        cosine_sum += float(
            (
                np.sum(bf_pred * qa_pred, axis=-1)
                / (np.linalg.norm(bf_pred, axis=-1) * np.linalg.norm(qa_pred, axis=-1) + 1e-45)
            ).sum()
        )

        bf_margin = np.sort(bf_pred, axis=-1)[:, -1] - np.sort(bf_pred, axis=-1)[:, -2]
        qa_margin = np.sort(qa_pred, axis=-1)[:, -1] - np.sort(qa_pred, axis=-1)[:, -2]
        margin_delta_sum += float(np.abs(bf_margin - qa_margin).sum())

        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  {index + 1}/{len(chunks)} chunks")

    mean_nll_bf16 = total_nll_bf16 / total_positions
    mean_nll_candidate = total_nll_candidate / total_positions
    results = {
        "bf16_dir": str(args.bf16_dir),
        "candidate_dir": str(args.candidate_dir),
        "corpus": str(args.corpus),
        "chunks": len(chunks),
        "tokens": total_positions,
        "bf16_ppl": math.exp(mean_nll_bf16),
        "candidate_ppl": math.exp(mean_nll_candidate),
        "delta_nll_per_token": mean_nll_candidate - mean_nll_bf16,
        "top1_agreement": top1_matches / total_positions,
        "top5_overlap": top5_overlap_sum / total_positions,
        "kl_bf16_candidate": kl_sum / total_positions,
        "cosine_logits": cosine_sum / total_positions,
        "mean_abs_top1_margin_delta": margin_delta_sum / total_positions,
    }
    print(json.dumps(results, indent=2))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
