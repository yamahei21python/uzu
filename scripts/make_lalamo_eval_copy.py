#!/usr/bin/env python3
"""Create a lalamo-loadable evaluation copy from a uzu-normalized model directory.

Conversion rewrites two things for uzu:
  * config.json (strict fields, no response_format)
  * HybridSpec metadata (flat form, original kept as model.safetensors.lalamo.bak)

lalamo evaluation (PPL / logits) needs the opposite form. This script links the
original weights and writes a lalamo config without duplicating tensor bytes.

Usage:
  python scripts/make_lalamo_eval_copy.py \
    --source /tmp/qwen35_0.8b_rht_int4_lalamo \
    --dest /tmp/qwen35_0.8b_rht_int4_lalamo_lalamo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

RESPONSE_FORMAT = {
    "reasoning_pattern": r"(?: response|(?:(?: think|(?=.* response))(?P<chain_of_thought>.*?)(?: response\s*|\Z)))?(?P<response>.*)",
    "prefixes": [" think", " response"],
    "turn_pattern": None,
    "tool_calls": {
        "tags": ["<tool_call>", "</tool_call>"],
        "function_pattern": r"<function=([^>\n]+)>(.*?)</function>",
        "parameter_pattern": r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>",
    },
}


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def lalamo_config(source: Path) -> dict:
    saved = source / "config.lalamo.json"
    if saved.exists():
        return json.loads(saved.read_text())

    config = json.loads((source / "config.json").read_text())
    token_codec = config.get("token_codec_config")
    if token_codec is not None:
        token_codec.pop("end_of_thinking_tag", None)
        token_codec.pop("output_parser_regex", None)
        token_codec["response_format"] = RESPONSE_FORMAT
    return config


def fresh_backup(weights: Path, backup: Path) -> bool:
    """The backup must not predate the weights it is supposed to mirror."""
    try:
        return backup.stat().st_mtime >= weights.stat().st_mtime
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "config.json").write_text(json.dumps(lalamo_config(args.source), indent=2) + "\n")

    weights = args.source / "model.safetensors"
    original_weights = args.source / "model.safetensors.lalamo.bak"
    use_backup = original_weights.exists() and fresh_backup(weights, original_weights)
    if original_weights.exists() and not use_backup:
        print(
            f"warning: {original_weights.name} is older than {weights.name}; "
            "ignoring the stale lalamo backup",
            flush=True,
        )
    link_or_copy(original_weights if use_backup else weights, args.dest / "model.safetensors")

    for name in ("tokenizer.json", "encoding.json"):
        source = args.source / name
        if source.exists():
            link_or_copy(source, args.dest / name)

    print(f"created {args.dest}")
    for file in sorted(args.dest.iterdir()):
        print(f"  {file.name}: {file.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
