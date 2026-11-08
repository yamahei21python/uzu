#!/usr/bin/env python3
"""CPU micro-benchmark for lalamo quantization operations.

Measures per-operation costs on CPU:
- RHT (Randomized Hadamard Transform, block sizes 32, 64, 128)
- Unpack (INT4 / INT8 to unpacked array)
- Pack (quantized array to packed format)
- Dequantize (group scale and zero-point application)
- Matmul (dense linear forward)
- Output RHT (output Hadamard transform)
- Calibration cost: H_I (G^T G) vs H_O (G G^T)

Can be executed standalone:
    python3 scripts/bench_quant_ops.py --repeats 20 --group-size 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def hadamard_transform_np(inputs: np.ndarray, block_size: int = 32) -> np.ndarray:
    """XLA-equivalent Hadamard transform using butterfly stages."""
    orig_shape = inputs.shape
    input_dim = orig_shape[-1]
    if input_dim % block_size != 0:
        raise ValueError(f"Input dimension {input_dim} must be divisible by block size {block_size}")

    result = inputs.reshape(-1, input_dim // block_size, block_size).copy()
    num_stages = block_size.bit_length() - 1

    for stage in range(num_stages):
        butterfly_size = 2 ** (stage + 1)
        half_butterfly_size = 2**stage
        *block_leading_dims, _ = result.shape
        grouped = result.reshape(
            *block_leading_dims,
            block_size // butterfly_size,
            butterfly_size,
        )
        left = grouped[..., :half_butterfly_size]
        right = grouped[..., half_butterfly_size:]
        result = np.concatenate((left + right, left - right), axis=-1).reshape(
            *block_leading_dims, block_size
        )

    normalization = np.sqrt(np.asarray(block_size, dtype=inputs.dtype))
    return (result / normalization).reshape(orig_shape)


def pack_uint_to_uint8_np(unpacked: np.ndarray, bits: int) -> np.ndarray:
    """Lalamo-equivalent pack_uint_to_uint8."""
    if bits == 8:
        return unpacked.astype(np.uint8)
    values_per_byte = 8 // bits
    *leading, last_dim = unpacked.shape
    grouped = unpacked.astype(np.uint8).reshape(*leading, last_dim // values_per_byte, values_per_byte)
    shifts = np.arange(values_per_byte, dtype=np.uint8) * np.uint8(bits)
    packed = np.sum(np.left_shift(grouped, shifts), axis=-1, dtype=np.uint8)
    return packed


def unpack_uint8_to_uint_np(
    packed: np.ndarray, bits: int, unpacked_last_axis_dim: int | None = None
) -> np.ndarray:
    """Lalamo-equivalent unpack_uint8_to_uint."""
    if bits == 8:
        if unpacked_last_axis_dim is None:
            return packed.astype(np.uint8)
        return packed[..., :unpacked_last_axis_dim].astype(np.uint8)

    values_per_byte = 8 // bits
    shifts = np.arange(values_per_byte, dtype=np.uint8) * np.uint8(bits)
    mask = np.uint8((1 << bits) - 1)
    unpacked = np.bitwise_and(np.right_shift(packed[..., None], shifts), mask)
    unpacked = unpacked.reshape(*packed.shape[:-1], -1)
    if unpacked_last_axis_dim is not None:
        unpacked = unpacked[..., :unpacked_last_axis_dim]
    return unpacked


def dequantize_np(
    int_weights: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray | None,
    group_size: int,
    bits: int,
) -> np.ndarray:
    """Lalamo-equivalent _integer_scaled_weights_to_master_weights."""
    *leading_dims, num_cols = int_weights.shape
    grouped_int_scale_weights = int_weights.reshape(
        *leading_dims,
        num_cols // group_size,
        group_size,
    )
    if zero_points is None:
        int_scale_zero_points = 2 ** (bits - 1)
    else:
        int_scale_zero_points = zero_points[..., None]

    grouped_weights = (grouped_int_scale_weights.astype(np.float32) - int_scale_zero_points) * scales[..., None]
    return grouped_weights.reshape(*leading_dims, num_cols)


def measure_op(fn, repeats: int = 20, warmup: int = 2) -> list[float]:
    """Measures execution times of fn across repeats, returning list of seconds."""
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        durations.append(t1 - t0)
    return durations


def summarize_times(durations: list[float]) -> dict[str, float]:
    arr = np.array(durations) * 1000.0  # ms
    return {
        "median_ms": float(np.median(arr)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "std_ms": float(np.std(arr)),
    }


def run_benchmark_for_shape(
    in_dim: int,
    out_dim: int,
    batch_size: int,
    group_size: int = 32,
    block_size: int = 32,
    bits: int = 4,
    has_output_block: bool = True,
    repeats: int = 20,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng(42)

    # Inputs & Weights setup
    x = rng.standard_normal((batch_size, in_dim), dtype=np.float32)
    signs_in = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(in_dim,))
    signs_out = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(out_dim,))

    # Quantized weight structures
    num_groups = in_dim // group_size
    packed_cols = in_dim if bits == 8 else (in_dim * bits + 7) // 8
    packed_weights = rng.integers(0, 256, size=(out_dim, packed_cols), dtype=np.uint8)
    scales = rng.uniform(0.01, 1.0, size=(out_dim, num_groups)).astype(np.float32)

    zp_packed_cols = num_groups if bits == 8 else (num_groups * bits + 7) // 8
    packed_zp = rng.integers(0, 256, size=(out_dim, zp_packed_cols), dtype=np.uint8)

    # 1. Input RHT: vector * signs -> hadamard_transform
    def op_input_rht():
        return hadamard_transform_np(x * signs_in, block_size=block_size)

    dur_input_rht = measure_op(op_input_rht, repeats=repeats)
    x_transformed = op_input_rht()

    # 2. INT Unpack: weights & zero_points
    def op_unpack():
        w_u = unpack_uint8_to_uint_np(packed_weights, bits=bits, unpacked_last_axis_dim=in_dim)
        zp_u = unpack_uint8_to_uint_np(packed_zp, bits=bits, unpacked_last_axis_dim=num_groups)
        return w_u, zp_u

    dur_unpack = measure_op(op_unpack, repeats=repeats)
    w_unpacked, zp_unpacked = op_unpack()

    # 2b. Pack (for reference/conversion analysis)
    def op_pack():
        return pack_uint_to_uint8_np(w_unpacked, bits=bits)

    dur_pack = measure_op(op_pack, repeats=repeats)

    # 3. Dequant: group scale and zero-point application
    def op_dequant():
        return dequantize_np(w_unpacked, scales, zp_unpacked, group_size=group_size, bits=bits)

    dur_dequant = measure_op(op_dequant, repeats=repeats)
    w_dequant = op_dequant()

    # 4. Matmul: x_transformed @ w_dequant.T
    def op_matmul():
        return x_transformed @ w_dequant.T

    dur_matmul = measure_op(op_matmul, repeats=repeats)
    y = op_matmul()

    # 5. Output RHT (if output block/incoherence transform is enabled)
    if has_output_block:
        def op_output_rht():
            return hadamard_transform_np(y, block_size=block_size) * signs_out

        dur_output_rht = measure_op(op_output_rht, repeats=repeats)
    else:
        dur_output_rht = [0.0] * repeats

    # 6. Full end-to-end forward call
    def op_total_forward():
        vec = hadamard_transform_np(x * signs_in, block_size=block_size)
        w_u = unpack_uint8_to_uint_np(packed_weights, bits=bits, unpacked_last_axis_dim=in_dim)
        zp_u = unpack_uint8_to_uint_np(packed_zp, bits=bits, unpacked_last_axis_dim=num_groups)
        w_dq = dequantize_np(w_u, scales, zp_u, group_size=group_size, bits=bits)
        out = vec @ w_dq.T
        if has_output_block:
            out = hadamard_transform_np(out, block_size=block_size) * signs_out
        return out

    dur_total = measure_op(op_total_forward, repeats=repeats)

    # Memory sizes in bytes
    packed_weight_bytes = packed_weights.nbytes
    scales_bytes = scales.nbytes
    zp_bytes = packed_zp.nbytes
    fp32_weight_bytes = out_dim * in_dim * 4
    total_compressed_bytes = packed_weight_bytes + scales_bytes + zp_bytes

    summary = {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "batch_size": batch_size,
        "group_size": group_size,
        "block_size": block_size,
        "bits": bits,
        "has_output_block": has_output_block,
        "repeats": repeats,
        "memory": {
            "packed_weight_bytes": packed_weight_bytes,
            "scales_bytes": scales_bytes,
            "zp_bytes": zp_bytes,
            "total_compressed_bytes": total_compressed_bytes,
            "fp32_weight_bytes": fp32_weight_bytes,
            "compression_ratio": float(fp32_weight_bytes / total_compressed_bytes),
        },
        "times": {
            "input_rht": summarize_times(dur_input_rht),
            "unpack": summarize_times(dur_unpack),
            "pack": summarize_times(dur_pack),
            "dequant": summarize_times(dur_dequant),
            "matmul": summarize_times(dur_matmul),
            "output_rht": summarize_times(dur_output_rht),
            "total_measured": summarize_times(dur_total),
        },
    }

    # Calculate breakdown percentages relative to sum of stages
    stage_sum = (
        summary["times"]["input_rht"]["median_ms"]
        + summary["times"]["unpack"]["median_ms"]
        + summary["times"]["dequant"]["median_ms"]
        + summary["times"]["matmul"]["median_ms"]
        + summary["times"]["output_rht"]["median_ms"]
    )
    summary["stage_sum_median_ms"] = stage_sum
    summary["percentages"] = {
        "input_rht": summary["times"]["input_rht"]["median_ms"] / max(stage_sum, 1e-9) * 100,
        "unpack": summary["times"]["unpack"]["median_ms"] / max(stage_sum, 1e-9) * 100,
        "dequant": summary["times"]["dequant"]["median_ms"] / max(stage_sum, 1e-9) * 100,
        "matmul": summary["times"]["matmul"]["median_ms"] / max(stage_sum, 1e-9) * 100,
        "output_rht": summary["times"]["output_rht"]["median_ms"] / max(stage_sum, 1e-9) * 100,
    }

    # Rank hotspots by median time
    stages = [
        ("unpack", summary["times"]["unpack"]["median_ms"]),
        ("dequant", summary["times"]["dequant"]["median_ms"]),
        ("matmul", summary["times"]["matmul"]["median_ms"]),
        ("output_rht", summary["times"]["output_rht"]["median_ms"]),
        ("input_rht", summary["times"]["input_rht"]["median_ms"]),
    ]
    stages.sort(key=lambda s: s[1], reverse=True)
    summary["bottleneck_ranking"] = [
        {"rank": idx + 1, "op": op, "median_ms": val, "percentage": val / max(stage_sum, 1e-9) * 100}
        for idx, (op, val) in enumerate(stages)
    ]

    return summary


def measure_calibration_ops(in_dim: int, out_dim: int, repeats: int = 5) -> dict[str, Any]:
    """Measures H_I and H_O accumulation for a given Linear layer shape."""
    rng = np.random.default_rng(42)
    G = rng.standard_normal((out_dim, in_dim), dtype=np.float32)

    def op_hi():
        return G.T @ G

    dur_hi = measure_op(op_hi, repeats=repeats, warmup=1)

    def op_ho():
        return G @ G.T

    dur_ho = measure_op(op_ho, repeats=repeats, warmup=1)

    hi_summary = summarize_times(dur_hi)
    ho_summary = summarize_times(dur_ho)

    h_i_bytes = in_dim * in_dim * 4
    h_o_bytes = out_dim * out_dim * 4

    return {
        "in_dim": in_dim,
        "out_dim": out_dim,
        "repeats": repeats,
        "h_i_median_ms": hi_summary["median_ms"],
        "h_o_median_ms": ho_summary["median_ms"],
        "h_i_memory_mb": h_i_bytes / (1024 * 1024),
        "h_o_memory_mb": h_o_bytes / (1024 * 1024),
        "h_o_to_h_i_ratio": ho_summary["median_ms"] / max(hi_summary["median_ms"], 1e-9),
    }


def main():
    parser = argparse.ArgumentParser(description="Micro-benchmark lalamo quantization ops on CPU")
    parser.add_argument("--repeats", type=int, default=20, help="Number of repetitions for median")
    parser.add_argument("--group-size", type=int, default=32, help="Quantization group size")
    parser.add_argument("--block-size", type=int, default=32, choices=(32, 64, 128), help="RHT block size")
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8), help="Quantization bits")
    parser.add_argument(
        "--json-out",
        type=str,
        default="research/lalamo_quant_hotspots.json",
        help="Path to output JSON",
    )
    args = parser.parse_args()

    shapes = [
        (1024, 1024),
        (1024, 8224),
        (2048, 1024),
        (1024, 7168),
        (7168, 1024),
    ]
    batches = [1, 128]

    # Load existing JSON if present to resume
    json_path = Path(args.json_out)
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"Resuming from existing {json_path}")
        except Exception:
            results = {}
    else:
        results = {}

    if "metadata" not in results:
        results["metadata"] = {
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cpu_info": os.popen("lscpu | grep 'Model name' || uname -m").read().strip(),
            "numpy_version": np.__version__,
            "repeats": args.repeats,
            "default_group_size": args.group_size,
            "default_block_size": args.block_size,
            "default_bits": args.bits,
        }
    for key in ["section_b_main", "calibration_ops", "section_c_group_sizes", "section_c_block_sizes", "section_c_output_block", "section_c_bits"]:
        if key not in results:
            results[key] = []

    def save_checkpoint():
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        mirror_path = Path("/workspace") / args.json_out
        if mirror_path.resolve() != json_path.resolve():
            mirror_path.parent.mkdir(parents=True, exist_ok=True)
            with open(mirror_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    def is_already_run(section_list, match_dict):
        for item in section_list:
            if all(item.get(k) == v for k, v in match_dict.items()):
                return True
        return False

    # Section B: 5 shapes x 2 batches with default settings
    print("\n--- Running Section B: Main Shapes x Batches ---")
    for in_dim, out_dim in shapes:
        for b in batches:
            match = {"in_dim": in_dim, "out_dim": out_dim, "batch_size": b, "group_size": args.group_size, "bits": args.bits}
            if is_already_run(results["section_b_main"], match):
                print(f"Skipping already computed in={in_dim}, out={out_dim}, batch={b}")
                continue
            print(f"Benchmarking in={in_dim}, out={out_dim}, batch={b} ...", flush=True)
            res = run_benchmark_for_shape(
                in_dim=in_dim,
                out_dim=out_dim,
                batch_size=b,
                group_size=args.group_size,
                block_size=args.block_size,
                bits=args.bits,
                has_output_block=True,
                repeats=args.repeats,
            )
            results["section_b_main"].append(res)
            save_checkpoint()

    # Calibration measurement: H_I vs H_O
    print("\n--- Running Calibration Simulation (H_I vs H_O) ---")
    for in_dim, out_dim in shapes:
        match = {"in_dim": in_dim, "out_dim": out_dim}
        if is_already_run(results["calibration_ops"], match):
            print(f"Skipping already computed calibration in={in_dim}, out={out_dim}")
            continue
        print(f"Measuring calib for in={in_dim}, out={out_dim} ...", flush=True)
        calib_res = measure_calibration_ops(in_dim, out_dim, repeats=5)
        results["calibration_ops"].append(calib_res)
        save_checkpoint()

    # Section C Variations on representative MLP layer (in=1024, out=7168, batch=128 & 1)
    print("\n--- Running Section C: Parameter Sensitivity Grid ---")
    rep_in, rep_out = 1024, 7168

    # 1. Group size: 32, 64, 128
    for gs in [32, 64, 128]:
        for b in [1, 128]:
            match = {"in_dim": rep_in, "out_dim": rep_out, "batch_size": b, "group_size": gs, "block_size": 32, "bits": 4}
            if is_already_run(results["section_c_group_sizes"], match):
                continue
            print(f"Testing group_size={gs}, batch={b} ...", flush=True)
            res = run_benchmark_for_shape(
                in_dim=rep_in,
                out_dim=rep_out,
                batch_size=b,
                group_size=gs,
                block_size=32,
                bits=4,
                has_output_block=True,
                repeats=args.repeats,
            )
            results["section_c_group_sizes"].append(res)
            save_checkpoint()

    # 2. Block size: 32, 64, 128
    for bs in [32, 64, 128]:
        for b in [1, 128]:
            match = {"in_dim": rep_in, "out_dim": rep_out, "batch_size": b, "group_size": 32, "block_size": bs, "bits": 4}
            if is_already_run(results["section_c_block_sizes"], match):
                continue
            print(f"Testing block_size={bs}, batch={b} ...", flush=True)
            res = run_benchmark_for_shape(
                in_dim=rep_in,
                out_dim=rep_out,
                batch_size=b,
                group_size=32,
                block_size=bs,
                bits=4,
                has_output_block=True,
                repeats=args.repeats,
            )
            results["section_c_block_sizes"].append(res)
            save_checkpoint()

    # 3. Output block: True vs False
    for ob in [True, False]:
        for b in [1, 128]:
            match = {"in_dim": rep_in, "out_dim": rep_out, "batch_size": b, "group_size": 32, "block_size": 32, "bits": 4, "has_output_block": ob}
            if is_already_run(results["section_c_output_block"], match):
                continue
            print(f"Testing output_block={ob}, batch={b} ...", flush=True)
            res = run_benchmark_for_shape(
                in_dim=rep_in,
                out_dim=rep_out,
                batch_size=b,
                group_size=32,
                block_size=32,
                bits=4,
                has_output_block=ob,
                repeats=args.repeats,
            )
            results["section_c_output_block"].append(res)
            save_checkpoint()

    # 4. Bits: 4 vs 8
    for bt in [4, 8]:
        for b in [1, 128]:
            match = {"in_dim": rep_in, "out_dim": rep_out, "batch_size": b, "group_size": 32, "block_size": 32, "bits": bt}
            if is_already_run(results["section_c_bits"], match):
                continue
            print(f"Testing bits={bt}, batch={b} ...", flush=True)
            res = run_benchmark_for_shape(
                in_dim=rep_in,
                out_dim=rep_out,
                batch_size=b,
                group_size=32,
                block_size=32,
                bits=bt,
                has_output_block=True,
                repeats=args.repeats,
            )
            results["section_c_bits"].append(res)
            save_checkpoint()

    save_checkpoint()
    print(f"\nAll benchmark results saved successfully!")


if __name__ == "__main__":
    main()
