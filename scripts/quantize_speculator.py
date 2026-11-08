#!/usr/bin/env python3
"""Quantize a converted DFlash speculator to int4 like the official Mirai draft.

`lalamo speculator convert` only writes float dtypes, so a converted draft stays
bf16 (about 2.2GB for the 9B draft) while the official one is 755MB. This
script reloads the draft as a lalamo module, compresses every Linear with a
4-bit group-32 hybrid spec (the same scheme as the target model) and writes the
result back in the uzu-loadable layout.

The speculator directory also needs `shapes.json`, which the converter does not
write; pass --shapes to copy one (or create an Argmax chain by default).

Usage:
  python scripts/quantize_speculator.py \
    --source /tmp/spec_guglxni --output /tmp/spec_guglxni_int4 \
    --bits 4 --group-size 32
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="converted speculator directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help="omit zero points (required for int8; uzu never compiles asymmetric int8)",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="skip the uzu metadata rewrite (the result will not load in uzu)",
    )
    parser.add_argument(
        "--shapes",
        type=Path,
        default=None,
        help="shapes.json to copy; default writes an Argmax chain with budget 16",
    )
    args = parser.parse_args()

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from lalamo.compressed.hybrid import HybridSpec
    from lalamo.compressed.incoherence import HadamardSpec, IncoherenceProcessingMode
    from lalamo.compressed.int import IntSpec
    from lalamo.models.speculator import SpeculatorModel
    from lalamo.modules.linear import Linear
    from lalamo.utils.sharding import ShardingConfig
    from lalamo.utils.surgery import map_nodes_of_type_with_path
    from lalamo.weight_matrix import Layout

    if args.bits == 8 and not args.symmetric:
        raise SystemExit("int8 must be symmetric: uzu only compiles zero-point dequant for 4 bits")

    sharding = ShardingConfig.replicated()
    started = time.time()
    model = SpeculatorModel.load(args.source, sharding, dtype=jnp.bfloat16)
    print(f"loaded {args.source} in {time.time() - started:.1f}s", flush=True)

    spec = HybridSpec(
        quantization_spec=IntSpec(
            bits=args.bits,
            group_size=args.group_size,
            is_symmetric=args.symmetric,
            layout=Layout.OUTPUT_INPUT,
        ),
        adapter_spec=None,
        incoherence_spec=None
        if args.block_size <= 0
        else HadamardSpec(
            block_size=args.block_size,
            processing_mode=IncoherenceProcessingMode.INPUT_OUTPUT,
        ),
    )

    quantized = 0
    skipped: list[tuple[str, tuple[int, ...]]] = []
    key = jax.random.key(0)

    def quantize_linear(path: tuple[str, ...], linear: Linear) -> Linear:
        nonlocal quantized
        shape = tuple(int(x) for x in linear.weights.shape)
        if len(shape) != 2 or any(dimension % args.group_size for dimension in shape):
            skipped.append(("/".join(path), shape))
            return linear
        started_at = time.time()
        new_weights = spec.compress(
            linear.weights.decompress(),
            key=key,
            sharding_config=linear.weights.sharding_config,
            is_sharded=linear.weights.is_sharded,
        )
        quantized += 1
        print(f"  {'.'.join(path)} {shape} {time.time() - started_at:.1f}s", flush=True)
        return eqx.tree_at(lambda x: x.weights, linear, new_weights)

    model = map_nodes_of_type_with_path(Linear, quantize_linear, model)
    print(f"quantized {quantized} Linear layers, skipped {len(skipped)}", flush=True)
    for path, shape in skipped:
        print(f"  skipped {path} {shape}")

    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    if args.shapes is not None:
        shutil.copy2(args.shapes, args.output / "shapes.json")
    else:
        (args.output / "shapes.json").write_text(
            json.dumps(
                {
                    "default": {
                        "tree_budget": 16,
                        "max_tree_depth": 16,
                        "dflash_depth_override": None,
                        "construction_method": {"type": "Argmax"},
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if not args.no_rewrite:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from rewrite_lalamo_hybrid_spec import rewrite

        # uzu only understands the flat HybridSpec form; lalamo writes the nested one.
        rewrite(args.output / "model.safetensors", backup=False)

    total = sum(path.stat().st_size for path in args.output.iterdir())
    print(f"wrote {args.output} ({total / 1e6:.0f}MB) in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
