from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jax")

ROOT = Path(__file__).resolve().parents[1]


def test_tiny_sharding_probe_on_two_cpu_devices() -> None:
    env = os.environ.copy()
    env["FORCE_2_CPU"] = "1"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "tiny_sharding_probe.py")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert len(report["devices"]) == 2
    names = {case["name"] for case in report["cases"]}
    assert names == {"column_sharded_matmul", "row_sharded_matmul", "replicated_reshard_matmul"}
