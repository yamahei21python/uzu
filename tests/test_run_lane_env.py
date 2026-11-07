from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lane import (  # noqa: E402
    Runner,
    find_dataset_dir,
    needs_triton_off,
    parse_compute_capabilities,
)


def test_parse_compute_capabilities_ignores_bad_lines() -> None:
    output = "7.5\n8.0\nnot-a-number\n7.0\n"
    assert parse_compute_capabilities(output) == [7.5, 8.0, 7.0]


def test_needs_triton_off_only_below_ampere() -> None:
    assert needs_triton_off([7.5, 7.5]) is True
    assert needs_triton_off([8.0]) is False
    assert needs_triton_off([8.9, 9.0]) is False
    assert needs_triton_off([]) is False


def make_model(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"x")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_find_dataset_dir_handles_both_mount_layouts(tmp_path: Path) -> None:
    old = tmp_path / "uzu-bf16-08b-lalamo"
    make_model(old)
    assert find_dataset_dir(tmp_path, "uzu-bf16-08b-lalamo") == old

    nested = tmp_path / "datasets" / "yamahei21python" / "uzu-bf16-08b-lalamo"
    make_model(nested)
    # remove the flat one so the nested layout is the only match
    (old / "model.safetensors").unlink()
    assert find_dataset_dir(tmp_path, "uzu-bf16-08b-lalamo") == nested


def test_find_dataset_dir_falls_back_to_any_model_with_tokenizer(tmp_path: Path) -> None:
    nested = tmp_path / "datasets" / "someone" / "renamed-dataset"
    make_model(nested)
    assert find_dataset_dir(tmp_path, "does-not-exist") == nested


def test_find_dataset_dir_returns_none_when_absent(tmp_path: Path) -> None:
    assert find_dataset_dir(tmp_path, "missing") is None


def make_runner(tmp_path: Path, spec_extra: dict) -> Runner:
    spec = {
        "lane": "t",
        "paths": {"kit": str(tmp_path), "work": str(tmp_path)},
        "conditions": [{"name": "a", "calibration_corpus": "c.txt"}],
    }
    spec["paths"].update(spec_extra)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return Runner(spec, spec_path, dry_run=True)


def test_environment_injects_spec_env_and_triton_flag(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, {"env": {"FOO": "bar"}, "jax_platform": "gpu"})
    runner.triton_off = True
    env = runner.environment(base={})
    assert env["FOO"] == "bar"
    assert env["JAX_PLATFORM_NAME"] == "gpu"
    assert "--xla_gpu_enable_triton_gemm=false" in env["XLA_FLAGS"]
    assert env["UZU_ATTENTION_IMPL"] == "xla"


def test_t4_forces_float32_eval_dtype(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, {"jax_platform": "gpu"})
    runner.triton_off = True
    runner.eval_dtype = "float32"
    assert runner.eval_dtype == "float32"


def test_ampere_keeps_bfloat16(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, {"jax_platform": "gpu"})
    runner.triton_off = False
    runner.eval_dtype = "bfloat16"
    assert runner.eval_dtype == "bfloat16"


def test_env_provenance_collects_numeric_knobs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("JAX_DEFAULT_MATMUL_PRECISION", "float32")
    monkeypatch.setenv("SOME_UNRELATED", "x")
    runner = make_runner(tmp_path, {"env": {"UZU_ATTENTION_IMPL": "xla"}})
    runner.triton_off = False
    provenance = runner.env_provenance()
    assert provenance["JAX_DEFAULT_MATMUL_PRECISION"] == "float32"
    assert provenance["UZU_ATTENTION_IMPL"] == "xla"
    assert "SOME_UNRELATED" not in provenance


def test_environment_keeps_existing_xla_flags_and_skips_triton_on_ampere(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, {"xla_flags": ["--xla_cpu_use_thunk_runtime=false"]})
    runner.triton_off = False
    env = runner.environment(base={"XLA_FLAGS": "--existing=1"})
    assert "--existing=1" in env["XLA_FLAGS"]
    assert "--xla_cpu_use_thunk_runtime=false" in env["XLA_FLAGS"]
    assert "--xla_gpu_enable_triton_gemm=false" not in env["XLA_FLAGS"]
    assert "UZU_ATTENTION_IMPL" not in env
