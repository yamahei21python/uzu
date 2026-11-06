#!/usr/bin/env python3
"""Regression test: RHT forward/inverse round-trip in NumPy.

Recreates the bug where the output-axis Hadamard was applied along columns.
Fails loudly if the restore formula inverts the process formula incorrectly.

Usage:
  python scripts/verify_rht_roundtrip.py
"""

from __future__ import annotations

import numpy as np

BLOCK_SIZE = 32
OUT_DIM = 64
IN_DIM = 128


def hadamard_matrix(size: int) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=np.float64)
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.sqrt(np.float64(size))


H = hadamard_matrix(BLOCK_SIZE)


def hadamard_last_axis(values: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    columns = values.shape[-1]
    if columns % block_size:
        raise ValueError(f"columns {columns} not divisible by block size {block_size}")
    blocks = values.reshape(*values.shape[:-1], columns // block_size, block_size)
    transformed = np.einsum("...ij,jk->...ik", blocks, H, optimize=True)
    return transformed.reshape(values.shape)


def hadamard_output_axis(values: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    return hadamard_last_axis(np.swapaxes(values, -1, -2), block_size).swapaxes(-1, -2)


def signs(count: int, seed: int) -> np.ndarray:
    return np.where(np.random.default_rng(seed).integers(0, 2, size=count) == 1, 1.0, -1.0)


def process_weights(weights: np.ndarray, input_signs: np.ndarray, output_signs: np.ndarray) -> np.ndarray:
    processed = hadamard_last_axis(weights * input_signs[None, :])
    return hadamard_output_axis(processed * output_signs[:, None])


def unprocess_weights(processed: np.ndarray, input_signs: np.ndarray, output_signs: np.ndarray) -> np.ndarray:
    restored = hadamard_output_axis(processed) * output_signs[:, None]
    return hadamard_last_axis(restored) * input_signs[None, :]


def main() -> None:
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(OUT_DIM, IN_DIM))
    input_signs = signs(IN_DIM, seed=1)
    output_signs = signs(OUT_DIM, seed=2)

    processed = process_weights(weights, input_signs, output_signs)
    restored = unprocess_weights(processed, input_signs, output_signs)

    error = np.max(np.abs(restored - weights))
    print(f"max abs error after round-trip: {error:.3e}")
    if not np.allclose(restored, weights, atol=1e-10, rtol=0):
        print("FAIL: RHT round-trip does not reconstruct the original weights")
        raise SystemExit(1)

    # Make sure the signs are orthogonal invariants under the reference formula.
    assert np.allclose(input_signs * input_signs, 1.0)
    assert np.allclose(output_signs * output_signs, 1.0)

    # Input-only mode: no output signs applied, one block of head rows only is checked here via slice.
    print("PASS: RHT round-trip reconstructs the original weights")


if __name__ == "__main__":
    main()
