#!/usr/bin/env python3
"""Compare two quantized checkpoints at the tensor-statistics level.

Useful for reverse-engineering a distributed checkpoint (e.g. Mirai-M) against
our own conversion:

  * scale mean / std / min / max
  * zero-point distribution
  * 4-bit code histogram
  * per-layer code agreement

Usage:
  python scripts/compare_quantized_checkpoints.py \
    --reference /tmp/mirai_08b_model/model.safetensors \
    --candidate /tmp/qwen35_0.8b_yaqa_hin_lalamo/model.safetensors \
    --json-out research/checkpoint_compare.json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    return header, 8 + header_length


def read_tensor(path: Path, header: dict, data_start: int, name: str) -> np.ndarray:
    info = header[name]
    start, end = info["data_offsets"]
    dtype = {"F32": np.float32, "BF16": np.uint16, "I32": np.int32, "U8": np.uint8}[info["dtype"]]
    with path.open("rb") as handle:
        handle.seek(data_start + start)
        raw = handle.read(end - start)
    return np.frombuffer(raw, dtype=dtype).reshape(info["shape"])


def hybrid_layers(path: Path) -> dict[str, dict]:
    header, data_start = read_header(path)
    layers = {}
    for key, value in header.get("__metadata__", {}).items():
        if not key.endswith(".weights.spec"):
            continue
        spec = json.loads(value)
        if spec.get("type") != "HybridSpec":
            continue
        base = key[: -len(".spec")]
        layers[base] = {
            "scales": read_tensor(path, header, data_start, f"{base}.quantized.scales").astype(np.float64),
            "zero_points": read_tensor(path, header, data_start, f"{base}.quantized.zero_points"),
            "weights": read_tensor(path, header, data_start, f"{base}.quantized.weights"),
        }
    return layers


def code_agreement(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return 0.0
    low_left = left & 0x0F
    low_right = right & 0x0F
    high_left = (left >> 4) & 0x0F
    high_right = (right >> 4) & 0x0F
    same = int((low_left == low_right).sum()) + int((high_left == high_right).sum())
    return same / (left.size * 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    reference = hybrid_layers(args.reference)
    candidate = hybrid_layers(args.candidate)
    common = sorted(set(reference) & set(candidate))
    if not common:
        raise SystemExit("no common HybridSpec layers")
    print(f"common layers: {len(common)}")

    rows = []
    code_histograms = {"reference": np.zeros(16, dtype=np.float64), "candidate": np.zeros(16, dtype=np.float64)}
    zero_points = {"reference": [], "candidate": []}
    for key in common:
        ref = reference[key]
        cand = candidate[key]
        agreement = code_agreement(ref["weights"], cand["weights"])
        rows.append(
            {
                "path": key,
                "scale_mean_reference": float(ref["scales"].mean()),
                "scale_mean_candidate": float(cand["scales"].mean()),
                "scale_ratio": float(ref["scales"].mean() / cand["scales"].mean())
                if cand["scales"].mean()
                else None,
                "code_agreement": agreement,
            }
        )
        for name, layer in (("reference", ref), ("candidate", cand)):
            packed = layer["weights"].reshape(-1)
            code_histograms[name] += np.bincount(packed & 0x0F, minlength=16)
            code_histograms[name] += np.bincount((packed >> 4) & 0x0F, minlength=16)
            zero_points[name].append(layer["zero_points"].reshape(-1))

    for name in code_histograms:
        total = code_histograms[name].sum()
        code_histograms[name] = code_histograms[name] / total if total else code_histograms[name]

    agreements = [row["code_agreement"] for row in rows]
    ratios = [row["scale_ratio"] for row in rows if row["scale_ratio"] is not None]
    zp_values = {
        name: np.concatenate([(values & 0x0F).astype(np.uint8) for values in chunks]
                             + [((values >> 4) & 0x0F).astype(np.uint8) for values in chunks])
        for name, chunks in zero_points.items()
    }

    summary = {
        "layers": len(common),
        "code_agreement_mean": float(np.mean(agreements)),
        "code_agreement_median": float(np.median(agreements)),
        "scale_ratio_mean": float(np.mean(ratios)) if ratios else None,
        "scale_ratio_median": float(np.median(ratios)) if ratios else None,
        "code_histogram_reference": code_histograms["reference"].round(6).tolist(),
        "code_histogram_candidate": code_histograms["candidate"].round(6).tolist(),
        "zero_point_mean_reference": float(zp_values["reference"].mean()),
        "zero_point_mean_candidate": float(zp_values["candidate"].mean()),
        "zero_point_std_reference": float(zp_values["reference"].std()),
        "zero_point_std_candidate": float(zp_values["candidate"].std()),
        "rows": rows,
    }

    print(f"code agreement: mean={summary['code_agreement_mean']:.4f} median={summary['code_agreement_median']:.4f}")
    print(f"scale ratio: mean={summary['scale_ratio_mean']:.6f} median={summary['scale_ratio_median']:.6f}")
    print(f"code histogram reference: {summary['code_histogram_reference']}")
    print(f"code histogram candidate: {summary['code_histogram_candidate']}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved {args.json_out}")


if __name__ == "__main__":
    main()
