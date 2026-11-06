#!/usr/bin/env python3
"""Attach an official DFlash speculator to a matching uzu model directory.

uzu auto-loads `model_dir/speculator/` when present. The speculator contains:
  config.json
  model.safetensors
  shapes.json

The target model must expose the same architecture the draft was trained on:
  * decoder hidden_dim == draft model_dim
  * decoder layer count >= max(target_layer_ids) + 1

Usage:
  python scripts/attach_speculator.py \
    --model-dir /tmp/huihui-9b-uzu \
    --repo trymirai/Qwen3.5-9B-M
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repo", default="trymirai/Qwen3.5-9B-M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-dir", type=Path, default=None)
    return parser.parse_args()


def decoder_config(model_dir: Path) -> dict:
    config = json.loads((model_dir / "config.json").read_text())
    return config["decoder_config"]["transformer_config"]


def main() -> None:
    args = parse_args()
    if not (args.model_dir / "config.json").exists():
        raise SystemExit(f"not an uzu model dir: {args.model_dir}")

    destination = args.model_dir / "speculator"
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"{destination} already exists; use --overwrite to replace")

    snapshot_root = args.download_dir or (args.model_dir.parent / ".speculator-download")
    snapshot_path = Path(
        snapshot_download(
            args.repo,
            revision=args.revision,
            allow_patterns=["speculator/*"],
            local_dir=str(snapshot_root),
        )
    )
    source = snapshot_path / "speculator"
    if not source.exists():
        raise SystemExit(f"no speculator/ in {args.repo}")

    draft_config = json.loads((source / "config.json").read_text())["speculator_config"]["draft_config"]
    target = decoder_config(args.model_dir)

    target_dim = int(target["model_dim"])
    target_layers = len(target["layer_configs"])
    draft_dim = int(draft_config["model_dim"])
    target_layer_ids = draft_config.get("target_layer_ids", [])
    num_target_layers = draft_config.get("num_target_layers")

    print(f"target: dim={target_dim} layers={target_layers}")
    print(f"draft : dim={draft_dim} target_layers={target_layer_ids}")
    if target_dim != draft_dim:
        raise SystemExit(f"hidden dim mismatch: target {target_dim}, draft {draft_dim}")
    if target_layer_ids and max(target_layer_ids) >= target_layers:
        raise SystemExit(f"draft targets layer {max(target_layer_ids)} but model has {target_layers} layers")
    if num_target_layers is not None and int(num_target_layers) != target_layers:
        raise SystemExit(f"layer count mismatch: target {target_layers}, draft num_target_layers {num_target_layers}")

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    print(f"attached {args.repo} -> {destination}")
    for file in sorted(destination.iterdir()):
        print(f"  {file.name}: {file.stat().st_size / 1e6:.1f} MB")
    print("Speculator will be loaded automatically by uzu on the next run.")


if __name__ == "__main__":
    main()
