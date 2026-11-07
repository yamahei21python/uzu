#!/usr/bin/env python3
"""Merge EN/JA raw Hessian calibrations with per-language normalization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--en", type=Path, required=True)
    parser.add_argument("--ja", type=Path, required=True)
    parser.add_argument("--ja-ratio", type=float, default=0.5)
    parser.add_argument("--damping", type=float, default=1e-3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def normalize(matrix: np.ndarray) -> np.ndarray:
    scale = float(np.mean(np.diagonal(matrix, axis1=-2, axis2=-1)))
    return matrix / max(scale, 1e-30)


def regularize(matrix: np.ndarray, damping: float) -> tuple[np.ndarray, float | None]:
    sigma = damping
    for _ in range(8):
        candidate = matrix + np.eye(matrix.shape[-1], dtype=matrix.dtype) * sigma
        try:
            np.linalg.cholesky(candidate)
            return candidate.astype(np.float32), sigma
        except np.linalg.LinAlgError:
            sigma *= 10.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-6, None)
    rebuilt = (eigenvectors * eigenvalues[..., None, :]) @ np.swapaxes(eigenvectors, -1, -2)
    return rebuilt.astype(np.float32), None


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.ja_ratio <= 1.0:
        raise SystemExit("--ja-ratio must be within [0, 1]")

    with np.load(args.en) as en_data, np.load(args.ja) as ja_data:
        en_keys = set(en_data.files)
        ja_keys = set(ja_data.files)
        keys = sorted(en_keys & ja_keys)
        if not keys:
            raise SystemExit("EN/JA calibration have no common keys")
        if en_keys != ja_keys:
            print(f"warning: key mismatch en_only={sorted(en_keys - ja_keys)} ja_only={sorted(ja_keys - en_keys)}")

        arrays: dict[str, np.ndarray] = {}
        records = []
        for key in keys:
            en = np.asarray(en_data[key], dtype=np.float64)
            ja = np.asarray(ja_data[key], dtype=np.float64)
            if en.shape != ja.shape:
                raise SystemExit(f"shape mismatch for {key}: {en.shape} vs {ja.shape}")
            mixed = (1.0 - args.ja_ratio) * normalize(en) + args.ja_ratio * normalize(ja)
            mixed, used_damping = regularize(mixed, args.damping)
            arrays[key] = mixed
            records.append(
                {
                    "key": key,
                    "shape": list(mixed.shape),
                    "damping": used_damping,
                    "en_trace": float(np.trace(normalize(en), axis1=-2, axis2=-1).sum()),
                    "ja_trace": float(np.trace(normalize(ja), axis1=-2, axis2=-1).sum()),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    np.savez(tmp, **arrays)
    os.replace(tmp, args.output)

    summary = {
        "en": str(args.en),
        "ja": str(args.ja),
        "ja_ratio": args.ja_ratio,
        "damping": args.damping,
        "arrays": len(arrays),
        "records": records,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"merged {len(arrays)} arrays -> {args.output}")


if __name__ == "__main__":
    main()
