#!/usr/bin/env python3
"""Compare uzu teacher-forced logits against lalamo JAX logits.

Input:
  --uzu-logits  raw little-endian f32 dump from
                `cargo run -p uzu-engine --example dump_logits -- <model> <out> <ids>`
  --lalamo-dir  lalamo-loadable model directory (raw config)
  --tokens      comma-separated token ids used for both engines

Output: JSON metrics on stdout (and optional --json-out).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uzu-logits", type=Path, required=True)
    parser.add_argument("--lalamo-dir", type=Path, required=True)
    parser.add_argument("--tokens", required=True, help="comma-separated token ids")
    parser.add_argument("--lalamo-path", default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lalamo_path:
        sys.path.insert(0, args.lalamo_path)

    import jax
    import jax.numpy as jnp
    from lalamo.model import Model
    from lalamo.module import Keychain
    from lalamo.modules import DecoderForwardPassConfig, ForwardPassMode
    from lalamo.utils.sharding import ShardingConfig

    tokens = [int(value) for value in args.tokens.split(",") if value.strip()]
    if not tokens:
        raise SystemExit("empty token list")

    raw = np.fromfile(args.uzu_logits, dtype="<f4")
    # uzu writes [len(tokens), vocab] without a header.
    if raw.size % len(tokens) != 0:
        raise SystemExit(f"uzu logits size {raw.size} is not divisible by {len(tokens)}")
    vocab = raw.size // len(tokens)
    uzu = raw.reshape(len(tokens), vocab).astype(np.float32)

    sharding = ShardingConfig.replicated()
    keychain = Keychain.init(seed=0, sharding_config=sharding)
    forward = DecoderForwardPassConfig.for_inference(mode=ForwardPassMode.MULTI_TOKEN)
    model = Model.load(args.lalamo_dir, sharding, dtype=jnp.bfloat16)

    ids = jnp.asarray([tokens], dtype=jnp.int32)
    positions = jnp.arange(len(tokens), dtype=jnp.int32)[None, :]
    result = model.decoder(
        token_ids=ids,
        token_positions=positions,
        state=None,
        return_updated_state=False,
        lengths_without_padding=None,
        forward_pass_config=forward,
        keychain=keychain.split(1)[0],
    )
    ref = np.asarray(result.logits[0], dtype=np.float32)

    if ref.shape != uzu.shape:
        raise SystemExit(f"shape mismatch: uzu={uzu.shape} lalamo={ref.shape}")

    top1_uzu = uzu.argmax(axis=-1)
    top1_ref = ref.argmax(axis=-1)
    top1 = float((top1_uzu == top1_ref).mean())

    top5_uzu = np.argpartition(uzu, -5, axis=-1)[:, -5:]
    top5_ref = np.argpartition(ref, -5, axis=-1)[:, -5:]
    top5 = float(np.mean([len(set(a) & set(b)) / 5.0 for a, b in zip(top5_uzu, top5_ref)]))

    def log_softmax(value: np.ndarray) -> np.ndarray:
        shifted = value - value.max(axis=-1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))

    log_uzu = log_softmax(uzu)
    log_ref = log_softmax(ref)
    # symmetric KL, mean over positions
    kl = float(((np.exp(log_uzu) * (log_uzu - log_ref)).sum(-1)
                + (np.exp(log_ref) * (log_ref - log_uzu)).sum(-1)).mean() / 2.0)

    cosine = float(np.mean(
        (uzu * ref).sum(-1) / (np.linalg.norm(uzu, axis=-1) * np.linalg.norm(ref, axis=-1) + 1e-12)
    ))
    delta = np.abs(uzu - ref)
    margin_uzu = np.take_along_axis(uzu, top1_uzu[:, None], axis=-1)[:, 0] - np.take_along_axis(
        uzu, top5_uzu[:, :1], axis=-1
    )[:, 0]
    margin_ref = np.take_along_axis(ref, top1_ref[:, None], axis=-1)[:, 0] - np.take_along_axis(
        ref, top5_ref[:, :1], axis=-1
    )[:, 0]

    metrics = {
        "positions": len(tokens),
        "vocab": int(vocab),
        "top1_agreement": top1,
        "top5_overlap": top5,
        "symmetric_kl": kl,
        "cosine_logits": cosine,
        "mean_abs_delta": float(delta.mean()),
        "max_abs_delta": float(delta.max()),
        "mean_abs_top1_margin_delta": float(np.abs(margin_uzu - margin_ref).mean()),
    }
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")


if __name__ == "__main__":
    main()
