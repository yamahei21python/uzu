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
        "eval": {"max_sequences": 10, "max_tokens": 64, "min_tokens": 16,
                 "corpora": {"en": "data/eval_en_wikitext2_dedup.txt"}},
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
    # the spec travels inside the script; there is no side file to lose
    assert not (out / "lane.json").exists()
    metadata = json.loads((out / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ["owner/uzu-bf16-demo"]
    code = (out / "uzu-lane-demo.py").read_text()
    assert "run_lane.py" in code
    assert "JAX_PLATFORM_NAME" in code
    assert 'LANE_SPEC = ' in code
    assert '"lane": "demo"' in code


def test_cpu_kernel_disables_gpu_and_pins_platform(tmp_path: Path) -> None:
    result = run(tmp_path, "--dataset", "owner/uzu-bf16-demo")
    assert result.returncode == 0, result.stderr
    out = tmp_path / "out"
    metadata = json.loads((out / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    code = (out / "uzu-lane-demo.py").read_text()
    assert "JAX_PLATFORM_NAME\"] = 'cpu'" in code
    assert "'--jax-platform', 'cpu'" in code


def test_reads_explicit_dataset_sources_from_spec(tmp_path: Path) -> None:
    spec = tmp_path / "lane.json"
    spec.write_text(json.dumps({
        "lane": "future-model",
        "model": {"bf16_source": "/kaggle/input/model"},
        "kaggle": {"dataset_sources": ["owner/model-9b"]},
        "conditions": [{"name": "only"}],
    }), encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(spec), "--kernel-id", "owner/future-model",
         "--out", str(tmp_path / "out")],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((tmp_path / "out" / "kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == ["owner/model-9b"]


def test_rejects_malformed_spec(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.json"
    malformed.write_text("[]", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--spec", str(malformed), "--kernel-id", "owner/demo",
         "--out", str(tmp_path / "out"), "--dataset", "not-an-id"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode != 0
    assert "invalid spec" in result.stderr + result.stdout


def test_rejects_malformed_dataset_id(tmp_path: Path) -> None:
    result = run(tmp_path, "--dataset", "not-an-id")
    assert result.returncode != 0
    assert "invalid dataset id" in result.stderr + result.stdout


def test_external_eval_requires_a_model_repo(tmp_path: Path) -> None:
    result = run(tmp_path, "--dataset", "owner/uzu-bf16-demo", "--kind", "external-eval")
    assert result.returncode != 0
    assert "--model-repo" in result.stderr + result.stdout


def test_external_eval_generates_download_and_corpora(tmp_path: Path) -> None:
    result = run(tmp_path, "--dataset", "owner/uzu-bf16-demo", "--kind", "external-eval",
                 "--model-repo", "trymirai/Qwen3.5-0.8B-M")
    assert result.returncode == 0, result.stderr
    code = (tmp_path / "out" / "uzu-lane-demo.py").read_text()
    assert "eval_external_model.py" in code
    assert "trymirai/Qwen3.5-0.8B-M" in code
    assert "eval_en_wikitext2_dedup.txt" in code
