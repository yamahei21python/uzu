#!/usr/bin/env python3
"""Measure the quantization error of every HybridMatrix (RHT + INT4) in a model.

Works directly on the exported safetensors:
  * codes      `...weights.quantized.weights`        uint8  (out, in / 2)
  * scales     `...weights.quantized.scales`         f32    (groups, out)
  * zero_points`...weights.quantized.zero_points`    uint8  (groups, out / 2)
  * input_signs`...weights.incoherence_signs.input_signs`   i32 (in,)
  * output_signs`...weights.incoherence_signs.output_signs` i32 (out,)

Dequantization reproduces lalamo `HadamardProcessor.unprocess_weights`:
  1. W = (code - zero_point) * scale                     (asymmetric INT4)
  2. output axis: W = hadamard(W) * output_signs
  3. input axis:  W = hadamard(W) * input_signs

Reference weights come from the unquantized lalamo save, same tensor path with
`...weights.weights`.

Usage:
  python scripts/verify_rht_int4_model.py \
    --quantized-dir /tmp/qwen35_0.8b_rht_int4_lalamo \
    --reference-dir /tmp/qwen35_0.8b_bf16_lalamo
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def load_tensor(path: Path, header: dict, data_start: int, name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    dtype = {
        "F32": np.float32,
        "BF16": np.uint16,
        "I32": np.int32,
        "U8": np.uint8,
    }[info["dtype"]]
    with path.open("rb") as f:
        f.seek(data_start + start)
        raw = f.read(end - start)
    array = np.frombuffer(raw, dtype=dtype).reshape(info["shape"])
    if info["dtype"] == "BF16":
        array = (array.astype(np.uint32) << 16).view(np.float32)
    return array


def hadamard_matrix(size: int) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=np.float32)
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(np.float32(size))


def hadamard_last_axis(values: np.ndarray, block_size: int) -> np.ndarray:
    leading = values.shape[:-1]
    rows, columns = values.reshape(-1, values.shape[-1]).shape[0], values.shape[-1]
    blocks = values.reshape(*leading, columns // block_size, block_size)
    matrix = hadamard_matrix(block_size)
    transformed = np.einsum("...ij,jk->...ik", blocks, matrix, optimize=True)
    return transformed.reshape(values.shape)


def unpack_int4(packed: np.ndarray, columns: int) -> np.ndarray:
    low = (packed & 0x0F).astype(np.float32)
    high = ((packed >> 4) & 0x0F).astype(np.float32)
    interleaved = np.stack((low, high), axis=-1)
    return interleaved.reshape(*packed.shape[:-1], columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantized-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.12,
        help="relative RMSE warning threshold (naive RTN INT4 group32 on Gaussian weights is ~0.08)",
    )
    parser.add_argument("--limit", type=int, default=0, help="only the first N tensors (debug)")
    parser.add_argument("--json-out", type=Path, default=None, help="save per-layer metrics as JSON")
    args = parser.parse_args()

    quant_path = args.quantized_dir / "model.safetensors"
    ref_path = args.reference_dir / "model.safetensors"
    quant_header, quant_start = read_header(quant_path)
    ref_header, ref_start = read_header(ref_path)

    weight_paths = sorted(
        key[: -len(".spec")]
        for key, value in quant_header.get("__metadata__", {}).items()
        if key.endswith(".weights.spec") and json.loads(value).get("type") == "HybridSpec"
    )
    if args.limit:
        weight_paths = weight_paths[: args.limit]
    print(f"hybrid weights: {len(weight_paths)}")

    total_squared = 0.0
    total_reference = 0.0
    failures = []
    rows = []
    for index, weight_path in enumerate(weight_paths):
        spec = json.loads(quant_header["__metadata__"][f"{weight_path}.spec"])
        quantization = spec["quantization_spec"]
        bits = quantization["bits"]
        group_size = quantization["group_size"]
        if bits != 4 or quantization["is_symmetric"]:
            raise SystemExit(f"unsupported spec at {weight_path}: {quantization}")

        quantized = f"{weight_path}.quantized"
        packed = load_tensor(quant_path, quant_header, quant_start, f"{quantized}.weights")
        scales = load_tensor(quant_path, quant_header, quant_start, f"{quantized}.scales").astype(np.float32)
        packed_zero_points = load_tensor(quant_path, quant_header, quant_start, f"{quantized}.zero_points")
        has_rht = f"{weight_path}.incoherence_signs.input_signs" in quant_header
        if has_rht:
            input_signs = load_tensor(
                quant_path, quant_header, quant_start, f"{weight_path}.incoherence_signs.input_signs"
            ).astype(np.float32)
            output_signs = load_tensor(
                quant_path, quant_header, quant_start, f"{weight_path}.incoherence_signs.output_signs"
            ).astype(np.float32)
        else:
            input_signs = None
            output_signs = None

        rows = scales.shape[-1]
        columns = packed.shape[-1] * 2
        codes = unpack_int4(packed, columns)

        zero_points = unpack_int4(packed_zero_points, rows)
        # (groups, rows) -> (1, groups, rows) broadcast against (rows, groups)
        groups = codes.shape[-1] // group_size
        grouped = codes.reshape(rows, groups, group_size)
        grouped_zero_points = zero_points.T.reshape(rows, groups, 1)
        grouped_scales = scales.T.reshape(rows, groups, 1)
        dequantized = (grouped - grouped_zero_points) * grouped_scales
        dequantized = dequantized.reshape(rows, columns)

        # output axis RHT restore, then input axis RHT restore
        if has_rht:
            dequantized = hadamard_last_axis(dequantized.T, args.block_size).T * output_signs[:, None]
            dequantized = hadamard_last_axis(dequantized, args.block_size) * input_signs[None, :]

        reference = load_tensor(ref_path, ref_header, ref_start, f"{weight_path}.weights").astype(np.float32)
        if reference.shape != dequantized.shape:
            failures.append((weight_path, f"shape {reference.shape} vs {dequantized.shape}"))
            continue

        error = dequantized - reference
        squared = float(np.sum(error * error, dtype=np.float64))
        norm = float(np.sum(reference * reference, dtype=np.float64))
        relative = (squared / norm) ** 0.5 if norm else float("inf")
        total_squared += squared
        total_reference += norm
        status = "OK" if relative <= args.threshold else "FAIL"
        if status == "FAIL":
            failures.append((weight_path, f"rel_rmse={relative:.6f}"))
        rows.append(
            {
                "path": weight_path,
                "shape": list(dequantized.shape),
                "weight_rmse": float(np.sqrt(squared / reference.size)),
                "relative_rmse": relative,
                "rht": has_rht,
            }
        )
        print(f"[{index + 1:3d}/{len(weight_paths)}] {status:4s} {relative:.6f}  {weight_path} {tuple(dequantized.shape)}")

    overall = (total_squared / total_reference) ** 0.5 if total_reference else float("inf")
    print(f"\noverall relative RMSE: {overall:.6f}")
    if args.json_out:
        args.json_out.write_text(
            json.dumps({"overall_relative_rmse": overall, "layers": rows}, indent=2) + "\n"
        )
        print(f"saved {args.json_out}")
    if failures:
        print(f"failures: {len(failures)}")
        for path, reason in failures:
            print(f"  {path}: {reason}")
        raise SystemExit(1)
    print("all layers within threshold")


if __name__ == "__main__":
    main()
