#!/usr/bin/env python3
"""Validate a single lalamo Linear through RHT + INT4 Group32.

Checks:
  1. compress a FullPrecision Linear to HybridMatrix
  2. measure output error
  3. export the HybridMatrix
  4. reload it through ShapeDtypeMatrix.load_exported
  5. confirm the reloaded output is identical to the quantized output

Usage:
  JAX_PLATFORM_NAME=cpu python scripts/validate_rht_int4_linear.py \
    --lalamo-path /tmp/lalamo-direct --in-dim 128 --out-dim 128
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lalamo-path", default=os.environ.get("LALAMO_PATH", "/tmp/lalamo-direct"))
    parser.add_argument("--in-dim", type=int, default=128)
    parser.add_argument("--out-dim", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32, choices=(32, 64, 128))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    sys.path.insert(0, args.lalamo_path)

    import jax
    import jax.numpy as jnp
    import equinox as eqx

    from lalamo.compressed.hybrid import HybridSpec
    from lalamo.compressed.int import IntSpec
    from lalamo.compressed.incoherence import HadamardSpec, IncoherenceProcessingMode
    from lalamo.initializer import RandomInitializer
    from lalamo.module import Keychain
    from lalamo.modules.linear import LinearConfig
    from lalamo.utils.dummy_array import dummy_array
    from lalamo.utils.sharding import ShardingConfig
    from lalamo.weight_matrix import Layout, ShapeDtypeSpec

    sharding = ShardingConfig.replicated()
    init = RandomInitializer(
        default_dtype=jnp.float32,
        key=jax.random.key(0),
        sharding_config=sharding,
    )
    linear = LinearConfig().init(
        initializer=init,
        input_dim=args.in_dim,
        output_dims=(args.out_dim,),
        has_biases=False,
        is_sharded=False,
    )
    x = jax.random.normal(jax.random.key(1), (args.in_dim,), dtype=jnp.float32) * 0.5
    keychain = Keychain.init(seed=42, sharding_config=sharding)
    y_orig = linear(x, keychain=keychain)[0]

    spec = HybridSpec(
        quantization_spec=IntSpec(
            bits=args.bits,
            group_size=args.group_size,
            is_symmetric=False,
            layout=Layout.OUTPUT_INPUT,
        ),
        adapter_spec=None,
        incoherence_spec=HadamardSpec(
            block_size=args.block_size,
            processing_mode=IncoherenceProcessingMode.INPUT_OUTPUT,
        ),
    )
    quantized_weights = spec.compress(
        linear.weights.decompress(),
        key=jax.random.key(2),
        sharding_config=linear.weights.sharding_config,
        is_sharded=linear.weights.is_sharded,
    )
    quantized_linear = eqx.tree_at(lambda m: m.weights, linear, quantized_weights)
    y_quantized = quantized_linear(x, keychain=keychain)[0]

    rmse = jnp.sqrt(jnp.mean((y_orig - y_quantized) ** 2))
    rel = rmse / (jnp.linalg.norm(y_orig) + 1e-12)
    print(f"shape: ({args.out_dim}, {args.in_dim})")
    print(f"RMSE: {float(rmse):.6f}")
    print(f"relative RMSE: {float(rel):.6f}")

    exported = quantized_weights.export()
    print("exported arrays:")
    for key, value in exported.arrays.items():
        print(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print("exported metadata:")
    for key, value in exported.metadata.items():
        print(f"  {key}: {value}")

    dummy_weights = dummy_array(
        (args.out_dim, args.in_dim),
        jnp.float32,
        sharding.resolve_sharding((None, None)),
    )
    dummy_matrix = ShapeDtypeSpec(layout=Layout.OUTPUT_INPUT).compress(
        dummy_weights,
        sharding_config=sharding,
        is_sharded=False,
    )
    loaded_weights = dummy_matrix.load_exported(exported)
    loaded_linear = eqx.tree_at(lambda m: m.weights, linear, loaded_weights)
    y_loaded = loaded_linear(x, keychain=keychain)[0]

    rmse_loaded = jnp.sqrt(jnp.mean((y_orig - y_loaded) ** 2))
    rmse_quantized_loaded = jnp.sqrt(jnp.mean((y_quantized - y_loaded) ** 2))
    print(f"loaded vs original RMSE: {float(rmse_loaded):.6f}")
    print(f"loaded vs quantized RMSE: {float(rmse_quantized_loaded):.6e}")
    if float(rmse_quantized_loaded) != 0.0:
        raise SystemExit("reload mismatch")


if __name__ == "__main__":
    main()
