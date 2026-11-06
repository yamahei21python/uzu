#!/usr/bin/env python3
"""Compare the full next-token distribution of a BF16 and an RHT+INT4 model.

Same token sequence -> same forward pass -> per-position comparison:

  Top-1 agreement   argmax(logits_bf16) == argmax(logits_int4)
  Top-5 overlap     |top5(bf16) ∩ top5(int4)| / 5
  KL(BF16 || INT4)  mean kl_div of the two distributions
  Cosine(logits)    cosine similarity of the raw logit vectors

The models are loaded with lalamo (not uzu) so the comparison measures the
quantization error itself and is independent of the generator's sampling loop.

Usage:
  JAX_PLATFORM_NAME=cpu python scripts/compare_logits.py \
    --bf16-dir /tmp/qwen35_0.8b_bf16_lalamo \
    --int4-dir /tmp/qwen35_0.8b_rht_int4_lalamo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-dir", type=Path, required=True)
    parser.add_argument("--int4-dir", type=Path, required=True)
    parser.add_argument("--lalamo-path", default=os.environ.get("LALAMO_PATH", "/tmp/lalamo-direct"))
    parser.add_argument("--max-positions", type=int, default=0, help="cap compared positions per prompt (0 = all)")
    parser.add_argument("--json-out", type=Path, default=None, help="save per-prompt metrics as JSON")
    return parser.parse_args()


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

    print("loading BF16 model...")
    bf16 = Model.load(args.bf16_dir, sharding, dtype=jnp.bfloat16)
    print("loading INT4 model...")
    int4 = Model.load(args.int4_dir, sharding, dtype=jnp.bfloat16)

    tokenizer = Tokenizer.from_file(str(args.bf16_dir / "tokenizer.json"))
    print("tokenizer loaded")

    prompts = [
        "The capital of Japan is",
        "1+1は？",
        "富士山の高さは",
        "def fibonacci(n):",
        "日本の首都は",
        "Translate to French: Hello",
        "Pythonで配列を逆順にするには",
        "ここでクイックソートは",
        "Explain in one sentence why the sky is blue:",
        "富士山の主な特徴は、",
    ]

    forward_pass_config = DecoderForwardPassConfig.for_inference(mode=ForwardPassMode.MULTI_TOKEN)

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

    def position_metrics(bf: np.ndarray, qa: np.ndarray) -> dict:
        bf_top1 = int(np.argmax(bf))
        qa_top1 = int(np.argmax(qa))
        bf_top5 = set(np.argsort(-bf)[:5])
        qa_top5 = set(np.argsort(-qa)[:5])
        bf_log = bf - bf.max()
        qa_log = qa - qa.max()
        bf_prob = np.exp(bf_log) / np.exp(bf_log).sum()
        qa_prob = np.exp(qa_log) / np.exp(qa_log).sum()
        kl = float(np.sum(bf_prob * (np.log(bf_prob + 1e-45) - np.log(qa_prob + 1e-45))))
        cosine = float(np.dot(bf, qa) / (np.linalg.norm(bf) * np.linalg.norm(qa) + 1e-45))
        return {
            "top1_match": int(bf_top1 == qa_top1),
            "top5_overlap": len(bf_top5 & qa_top5) / 5.0,
            "bf16_top1_in_int4_top5": int(bf_top1 in qa_top5),
            "kl_bf16_int4": kl,
            "cosine_logits": cosine,
        }

    rows = []
    positions_total = 0
    matched = 0
    overlap_sum = 0.0
    kl_sum = 0.0
    cosine_sum = 0.0

    for index, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_special_tokens=True).ids
        bf_logits = logits_for(bf16, ids)
        qa_logits = logits_for(int4, ids)
        if args.max_positions:
            bf_logits = bf_logits[-args.max_positions:]
            qa_logits = qa_logits[-args.max_positions:]
        per_position = [position_metrics(bf, qa) for bf, qa in zip(bf_logits, qa_logits)]
        top1 = sum(m["top1_match"] for m in per_position)
        top5 = float(np.mean([m["top5_overlap"] for m in per_position]))
        kl = float(np.mean([m["kl_bf16_int4"] for m in per_position]))
        cosine = float(np.mean([m["cosine_logits"] for m in per_position]))
        rows.append(
            {
                "prompt": prompt,
                "positions": len(per_position),
                "top1_matches": top1,
                "top5_overlap": top5,
                "kl_bf16_int4": kl,
                "cosine_logits": cosine,
            }
        )
        positions_total += len(per_position)
        matched += top1
        overlap_sum += sum(m["top5_overlap"] for m in per_position)
        kl_sum += sum(m["kl_bf16_int4"] for m in per_position)
        cosine_sum += sum(m["cosine_logits"] for m in per_position)
        print(
            f"[{index + 1:2d}] positions={len(per_position):3d} "
            f"top1={top1}/{len(per_position)} "
            f"top5={top5:.2f} "
            f"kl={kl:.4f} "
            f"cos={cosine:.6f}  {prompt!r}"
        )

    print(f"\nper-position metrics over {positions_total} positions")
    print(f"  Top-1 agreement : {matched / positions_total:.3f}")
    print(f"  Top-5 overlap   : {overlap_sum / positions_total:.3f}")
    print(f"  KL(BF16||INT4)  : {kl_sum / positions_total:.4f}")
    print(f"  Cosine(logits)  : {cosine_sum / positions_total:.6f}")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
        print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
