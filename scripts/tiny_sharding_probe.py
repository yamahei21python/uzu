#!/usr/bin/env python3
"""Tiny 2-device sharding probe for uzu/lalamo TP semantics.

Usage on Kaggle T4x2:
  python scripts/tiny_sharding_probe.py

Usage on CPU with 2 forced devices:
  FORCE_2_CPU=1 python scripts/tiny_sharding_probe.py
"""

from __future__ import annotations

import json
import os

if os.environ.get("FORCE_2_CPU") == "1":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P


def main() -> None:
    devices = jax.devices()
    report = {"devices": [str(device) for device in devices], "cases": [], "ok": False}
    if len(devices) < 2:
        report["error"] = f"need >=2 devices, got {len(devices)}"
        print(json.dumps(report, indent=2))
        raise SystemExit(1)

    mesh = Mesh(np.asarray(devices[:2]), ("tensor",), axis_types=(AxisType.Explicit,))
    base = jnp.asarray(np.arange(64, dtype=np.float32).reshape(8, 8))

    def record(name: str, expect_error: bool, expected: str, fn) -> None:
        try:
            fn()
        except Exception as error:  # noqa: BLE001 - probe records the exact runtime error
            message = str(error)
            passed = expect_error and expected in message
            report["cases"].append(
                {"name": name, "expected": expected, "ok": passed, "error": message[:500]}
            )
        else:
            report["cases"].append(
                {"name": name, "expected": expected, "ok": not expect_error, "error": None}
            )

    def column_sharded_matmul() -> None:
        sharding = NamedSharding(mesh, P(None, "tensor"))
        with jax.set_mesh(mesh):
            x = jax.device_put(base, sharding)
            _ = x @ x.T

    def row_sharded_matmul() -> None:
        sharding = NamedSharding(mesh, P("tensor", None))
        with jax.set_mesh(mesh):
            x = jax.device_put(base, sharding)
            _ = x @ x.T

    def replicated_reshard_matmul() -> None:
        sharded = NamedSharding(mesh, P(None, "tensor"))
        replicated = NamedSharding(mesh, P())
        with jax.set_mesh(mesh):
            x = jax.device_put(base, sharded)
            x_rep = jax.device_put(x, replicated)
            result = x_rep @ x_rep.T
            expected = base @ base.T
            diff = float(jnp.max(jnp.abs(result - expected)))
            report["reshard_max_abs_diff"] = diff
            if diff > 1e-4:
                raise RuntimeError(f"reshard mismatch: {diff}")

    record("column_sharded_matmul", True, "Contracting dimensions are sharded", column_sharded_matmul)
    record("row_sharded_matmul", True, "illegally sharded", row_sharded_matmul)
    record("replicated_reshard_matmul", False, "", replicated_reshard_matmul)

    report["ok"] = all(case["ok"] for case in report["cases"])
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
