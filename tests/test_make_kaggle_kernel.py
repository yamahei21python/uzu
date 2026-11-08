"""A kernel generated from a lane spec must be self-contained and pushable."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_kaggle_kernel.py"


def run(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    spec = tmp_path / "lane.json"
    spec.write_text(json.dumps({
        "lane": "demo",
        "model": {"bf16_source": "/kaggle/input/uzu-bf16-demo"},
        "paths": {"kit": "kit", "work": "/kaggle/working"},
        "eval": {"max_sequences": 10, "max_tokens": 64, "min_tokens": 16, "corpora": {}},
        "conditions": [{"name": "only", "samples": 8}],
    }), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec), "--kernel-id", "owner/uzu-lane-demo",
         "--out", str(tmp_path / "out"), *extra],
        text=True, capture_output=True, check=False,
    )


def test_requires_a_dataset_source(tmp_path: Path) -> None:
    result = run(tmp_path)
    assert result.returncode != 0
    assert "--dataset" in result.stderr + result.stdout


def test_generates_kernel_and_metadata(tmp_path: Path) -> None:
    result = run(tmp_path, "--dataset", "owner/uzu-bf16-demo", "--platform", "gpu")
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    assert (out / "uzu-lane-demo.py").exists()
    assert (out / "lane.json").exists()
    metadata = json.loads((out / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ["owner/uzu-bf16-demo"]
    code = (out / "uzu-lane-demo.py").read_text()
    assert "run_lane.py" in code
    assert "JAX_PLATFORM_NAME" in code


def test_cpu_kernel_disables_gpu_and_pins_platform(tmp_path: Path) -> None:
    result = run(tmp_path, "--dataset", "owner/uzu-bf16-demo")
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    metadata = json.loads((out / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    code = (out / "uzu-lane-demo.py").read_text()
    assert "JAX_PLATFORM_NAME\"] = 'cpu'" in code
    assert "'--jax-platform', 'cpu'" in code
