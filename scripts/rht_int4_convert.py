#!/usr/bin/env python3
"""Convert a supported Hugging Face model to uzu with RHT + INT4 Group32.

This bypasses MLX quantization and applies lalamo's HybridSpec directly:
  BF16 Linear -> RHT (Hadamard block 32) -> asymmetric INT4 group 32 -> HybridMatrix

The resulting lalamo model is saved, its HybridSpec metadata is rewritten to
uzu's flat form, and encoding.json is added.

Example (0.8B, local CPU):
  JAX_PLATFORM_NAME=cpu python scripts/rht_int4_convert.py \
    --spec-name Qwen3.5-0.8B \
    --lalamo-path /tmp/lalamo-direct \
    --output-dir /tmp/qwen35_0.8b_rht_int4_lalamo \
    --bf16-dir /tmp/qwen35_0.8b_bf16_lalamo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-name", default="Qwen3.5-0.8B", help="Built-in lalamo model spec name")
    parser.add_argument("--hf-repo", default=None, help="Override origin with this Hugging Face repo")
    parser.add_argument("--vendor", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bf16-dir", default=None, help="Also save the unquantized model here")
    parser.add_argument(
        "--load-from",
        default=None,
        help="Load an existing lalamo model directory instead of importing from Hugging Face",
    )
    parser.add_argument("--lalamo-path", default=os.environ.get("LALAMO_PATH", "/tmp/lalamo-direct"))
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32, choices=(32, 64, 128))
    parser.add_argument(
        "--no-rht",
        action="store_true",
        help="Disable RHT and quantize with plain INT4 Group (isolation baseline)",
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--skip-incompatible", action="store_true", help="Leave non-divisible Linears full precision")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--preconditioners",
        default=None,
        help="NPZ from scripts/yaqa_calibration.py; keys are '<path>/input_block' and optional '/output_block'",
    )
    return parser.parse_args()


# uzu config strictness:
#   * every struct uses `#[serde(deny_unknown_fields)]`
#   * every field is required, even `Option<T>` (missing != null)
# lalamo emits a newer token_codec schema that violates both rules.
_UZU_TOKEN_CODEC_REQUIRED = ("output_parser_regex", "end_of_thinking_tag")
_UZU_TOKEN_CODEC_UNSUPPORTED = ("response_format", "reasoning_effort")


def normalize_config_for_uzu(output_dir: Path) -> None:
    path = output_dir / "config.json"
    config = json.loads(path.read_text())

    # Keep the untouched lalamo form for lalamo-side evaluation (PPL / logits).
    lalamo_path = output_dir / "config.lalamo.json"
    if not lalamo_path.exists():
        lalamo_path.write_text(json.dumps(config, indent=2) + "\n")

    token_codec = config.get("token_codec_config")
    if token_codec is None:
        return

    removed = [key for key in _UZU_TOKEN_CODEC_UNSUPPORTED if token_codec.pop(key, None) is not None]
    added = [key for key in _UZU_TOKEN_CODEC_REQUIRED if key not in token_codec]
    for key in added:
        token_codec[key] = None

    if removed or added:
        path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"normalize config.json: removed={removed or '[]'} added={added or '[]'}")


def find_spec(qwen_models, name: str):
    for spec in qwen_models:
        if spec.name == name:
            return spec
    raise ValueError(f"model spec {name!r} not found")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, args.lalamo_path)

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import numpy as np

    from lalamo.compressed.hybrid import HybridSpec
    from lalamo.compressed.int import IntSpec
    from lalamo.compressed.incoherence import HadamardSpec, IncoherenceProcessingMode
    from lalamo.model_import.common import import_model
    from lalamo.model_import.model_specs.qwen import QWEN_MODELS
    from lalamo.modules.linear import Linear
    from lalamo.preconditioner import Preconditioner
    from lalamo.utils.sharding import ShardingConfig
    from lalamo.utils.surgery import map_nodes_of_type_with_path, select_nodes_of_type
    from lalamo.weight_matrix import Layout
    from rewrite_lalamo_hybrid_spec import rewrite

    if args.load_from:
        from lalamo.model import Model

        print(f"load lalamo model <- {args.load_from}")
        model = Model.load(args.load_from, sharding=ShardingConfig.replicated(), dtype=jnp.dtype(args.dtype))
        print("load done")
    else:
        spec = find_spec(QWEN_MODELS, args.spec_name)
        if args.hf_repo:
            from lalamo.model_import.origins import HuggingFaceOrigin

            spec = replace(
                spec,
                vendor=args.vendor or spec.vendor,
                name=args.name or spec.name,
                origin=HuggingFaceOrigin(repo=args.hf_repo),
            )

        print(f"import {spec.vendor}/{spec.name} <- {spec.origin.description}")
        results = import_model(
            spec,
            sharding_config=ShardingConfig.replicated(),
            dtype=jnp.dtype(args.dtype),
        )
        model = results.model
        print("import done")

    if args.bf16_dir:
        bf16_dir = Path(args.bf16_dir)
        bf16_dir.mkdir(parents=True, exist_ok=True)
        print(f"save BF16 baseline -> {bf16_dir}")
        model.save(bf16_dir)
        normalize_config_for_uzu(bf16_dir)
        print("BF16 baseline saved")

    linear_list = select_nodes_of_type(Linear, model)
    required_divisor = args.group_size if args.no_rht else max(args.block_size, args.group_size)
    bad = []
    for path, linear in linear_list:
        shape = tuple(int(x) for x in linear.weights.shape)
        if len(shape) != 2 or any(dimension % required_divisor for dimension in shape):
            bad.append(("/".join(path), shape))

    print(f"Linear layers: {len(linear_list)}, incompatible: {len(bad)}")
    for path, shape in bad:
        print(f"  {path}: {shape}")
    if bad and not args.skip_incompatible:
        raise SystemExit("incompatible Linear shapes; use --skip-incompatible to leave them full precision")

    hybrid_spec = HybridSpec(
        quantization_spec=IntSpec(
            bits=args.bits,
            group_size=args.group_size,
            is_symmetric=False,
            layout=Layout.OUTPUT_INPUT,
        ),
        adapter_spec=None,
        incoherence_spec=None
        if args.no_rht
        else HadamardSpec(
            block_size=args.block_size,
            processing_mode=IncoherenceProcessingMode.INPUT_OUTPUT,
        ),
    )
    quantize_key = jax.random.key(0)
    counts = {"quantized": 0, "skipped": 0, "yaqa": 0}

    calibration = None
    if args.preconditioners:
        calibration = np.load(args.preconditioners)
        print(f"preconditioners: {args.preconditioners} ({len(calibration.files)} arrays)")

    def preconditioner_for(path) -> Preconditioner | None:
        if calibration is None:
            return None
        key = "/".join(path)
        input_key = f"{key}/input_block"
        if input_key not in calibration:
            return None
        output_key = f"{key}/output_block"
        return Preconditioner.init(
            input_block=jnp.asarray(calibration[input_key], dtype=jnp.float32),
            output_block=jnp.asarray(calibration[output_key], dtype=jnp.float32)
            if output_key in calibration
            else None,
        )

    def quantize_linear(path, linear: Linear) -> Linear:
        shape = tuple(int(x) for x in linear.weights.shape)
        if len(shape) != 2 or any(dimension % required_divisor for dimension in shape):
            counts["skipped"] += 1
            if not args.skip_incompatible:
                raise ValueError(f"incompatible Linear {path}: {shape}")
            return linear
        counts["quantized"] += 1
        preconditioner = preconditioner_for(path)
        if preconditioner is not None:
            counts["yaqa"] += 1
        new_weights = hybrid_spec.compress(
            linear.weights.decompress(),
            key=quantize_key,
            preconditioner=preconditioner,
            sharding_config=linear.weights.sharding_config,
            is_sharded=linear.weights.is_sharded,
        )
        return eqx.tree_at(lambda x: x.weights, linear, new_weights)

    if args.dry_run:
        print("dry-run: no quantization performed")
        return

    if args.no_rht:
        print(f"quantize -> plain INT{args.bits} group {args.group_size} (symmetric=False, layout=output_input)")
    else:
        print(
            f"quantize -> RHT block {args.block_size} + INT{args.bits} group {args.group_size} "
            f"(symmetric=False, layout=output_input)"
        )
    quantized_model = map_nodes_of_type_with_path(Linear, quantize_linear, model)
    print(f"quantized: {counts['quantized']}, skipped: {counts['skipped']}, yaqa: {counts['yaqa']}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"save quantized model -> {output_dir}")
    quantized_model.save(output_dir)

    print("rewrite HybridSpec metadata for uzu")
    rewrite(output_dir / "model.safetensors")

    normalize_config_for_uzu(output_dir)

    encoding = {"type": "hanashi", "name": "qwen3.5"}
    (output_dir / "encoding.json").write_text(json.dumps(encoding, indent=2) + "\n")
    print("encoding.json written")

    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
