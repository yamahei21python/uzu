from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_qad_dataset import validate_student  # noqa: E402


def make_model(root: Path, name: str, *, response_format: bool = True, overlap: bool = True, tokenizer: str = "tok") -> Path:
    model = root / name
    model.mkdir(parents=True)
    token_codec = {"eos_token": "<eos>"}
    if response_format:
        token_codec["response_format"] = {"prefixes": [" think"]}
    (model / "config.json").write_text(json.dumps({"token_codec_config": token_codec}), encoding="utf-8")
    (model / "tokenizer.json").write_text(tokenizer, encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "provenance.json").write_text(json.dumps({"kind": "yaqa", "overlap_checked": overlap}), encoding="utf-8")
    return model


def test_valid_student_passes(tmp_path: Path) -> None:
    bf16 = make_model(tmp_path, "bf16")
    student = make_model(tmp_path, "student")
    assert validate_student(bf16, student) == []


def test_normalized_config_is_rejected(tmp_path: Path) -> None:
    bf16 = make_model(tmp_path, "bf16")
    student = make_model(tmp_path, "student", response_format=False)
    problems = validate_student(bf16, student)
    assert any("uzu-normalized" in problem for problem in problems)


def test_unchecked_provenance_is_rejected(tmp_path: Path) -> None:
    bf16 = make_model(tmp_path, "bf16")
    student = make_model(tmp_path, "student", overlap=False)
    problems = validate_student(bf16, student)
    assert any("overlap_checked=false" in problem for problem in problems)


def test_tokenizer_mismatch_is_rejected(tmp_path: Path) -> None:
    bf16 = make_model(tmp_path, "bf16", tokenizer="a")
    student = make_model(tmp_path, "student", tokenizer="b")
    problems = validate_student(bf16, student)
    assert any("tokenizer.json differs" in problem for problem in problems)


def test_missing_provenance_is_rejected(tmp_path: Path) -> None:
    bf16 = make_model(tmp_path, "bf16")
    student = make_model(tmp_path, "student")
    (student / "provenance.json").unlink()
    problems = validate_student(bf16, student)
    assert any("provenance.json missing" in problem for problem in problems)


def test_dry_run_lists_the_three_steps(tmp_path: Path) -> None:
    bf16 = make_model(tmp_path, "bf16")
    root = tmp_path / "work"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_qad_dataset.py"),
            "--bf16",
            str(bf16),
            "--out",
            str(root),
            "--corpus",
            "calibration.txt",
            "--eval-corpus",
            "eval_en.txt",
            "--eval-corpus",
            "eval_ja.txt",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "yaqa_calibration.py" in result.stdout
    assert "rht_int4_convert.py" in result.stdout
    assert "make_lalamo_eval_copy.py" in result.stdout
    assert result.stdout.count("--eval-corpus") == 2
    assert not (root / "student").exists()
