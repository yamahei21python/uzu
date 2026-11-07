from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_corpus_guard_rejects_overlap(tmp_path: Path) -> None:
    eval_corpus = tmp_path / "eval.txt"
    calib = tmp_path / "calib.txt"
    eval_corpus.write_text("this is a long evaluation line\n", encoding="utf-8")
    calib.write_text("this is a long evaluation line\n", encoding="utf-8")
    result = run(
        "corpus_guard.py",
        "--calibration",
        str(calib),
        "--eval",
        str(eval_corpus),
    )
    assert result.returncode == 1
    assert "overlaps" in result.stderr


def test_corpus_guard_accepts_disjoint(tmp_path: Path) -> None:
    eval_corpus = tmp_path / "eval.txt"
    calib = tmp_path / "calib.txt"
    eval_corpus.write_text("evaluation only line with enough length\n", encoding="utf-8")
    calib.write_text("calibration only line with enough length\n", encoding="utf-8")
    result = run(
        "corpus_guard.py",
        "--calibration",
        str(calib),
        "--eval",
        str(eval_corpus),
    )
    assert result.returncode == 0


def test_dedup_lines_removes_duplicates(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "output.txt"
    source.write_text("a\nb\na\nc\nb\n", encoding="utf-8")
    result = run("dedup_lines.py", "--input", str(source), "--output", str(output))
    assert result.returncode == 0
    assert output.read_text(encoding="utf-8").splitlines() == ["a", "b", "c"]


def _write_config(tmp_path: Path, *, overlap: bool = False) -> tuple[Path, Path]:
    eval_en = tmp_path / "eval_en.txt"
    eval_ja = tmp_path / "eval_ja.txt"
    calib = tmp_path / "calib.txt"
    eval_en.write_text(
        "the quick brown fox jumps over the lazy dog\n"
        "a second distinct evaluation sentence for the fixture\n"
        "a third distinct evaluation sentence for the fixture\n",
        encoding="utf-8",
    )
    eval_ja.write_text(
        "日本語の評価用の長い文章です。これは十分な長さがあります。\n"
        "二つ目の評価用文章です。これも十分な長さがあります。\n"
        "三つ目の評価用文章です。これも十分な長さがあります。\n",
        encoding="utf-8",
    )
    if overlap:
        calib.write_text("the quick brown fox jumps over the lazy dog\n", encoding="utf-8")
    else:
        calib.write_text("a completely separate calibration sentence for testing\n", encoding="utf-8")
    config = {
        "name": "fixture",
        "model": {},
        "eval": {
            "max_sequences": 2,
            "max_tokens": 4,
            "min_tokens": 1,
            "corpora": {"en": str(eval_en), "ja": str(eval_ja)},
        },
        "calibration": {
            "samples": 2,
            "max_tokens": 4,
            "output_block": "small",
            "damping": 0.001,
            "corpora": {"calib": str(calib)},
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, calib


def _write_ppl(root: Path, corpus: Path, chunks: int = 2) -> None:
    directory = root / "condition"
    directory.mkdir()
    payload = {
        "corpus": str(corpus),
        "chunks": chunks,
        "tokens": chunks * 4 - chunks,
        "bf16_ppl": 1.0,
        "candidate_ppl": 1.1,
    }
    (directory / "ppl_en.json").write_text(json.dumps(payload), encoding="utf-8")


def test_check_experiment_accepts_consistent_fixture(tmp_path: Path) -> None:
    config, calib = _write_config(tmp_path)
    eval_en = tmp_path / "eval_en.txt"
    _write_ppl(tmp_path, eval_en)
    (tmp_path / "condition" / "calibration.json").write_text(
        json.dumps(
            {
                "corpus": str(calib),
                "samples": 2,
                "max_tokens": 4,
                "output_block": "small",
                "damping": 0.001,
            }
        ),
        encoding="utf-8",
    )
    result = run("check_experiment.py", "--config", str(config), "--root", str(tmp_path))
    assert result.returncode == 0, result.stdout + result.stderr


def test_check_experiment_rejects_chunk_drift(tmp_path: Path) -> None:
    config, calib = _write_config(tmp_path)
    eval_en = tmp_path / "eval_en.txt"
    _write_ppl(tmp_path, eval_en, chunks=3)
    result = run("check_experiment.py", "--config", str(config), "--root", str(tmp_path))
    assert result.returncode == 1
    assert "chunks=3" in result.stdout


def test_check_experiment_rejects_corpus_overlap(tmp_path: Path) -> None:
    config, _ = _write_config(tmp_path, overlap=True)
    eval_en = tmp_path / "eval_en.txt"
    _write_ppl(tmp_path, eval_en)
    result = run("check_experiment.py", "--config", str(config), "--root", str(tmp_path))
    assert result.returncode == 1
    assert "corpus overlap" in result.stdout


def test_merge_calibration_groups_validates_coverage(tmp_path: Path) -> None:
    def write_group(name: str, layer: int) -> Path:
        path = tmp_path / name
        np.savez(
            path,
            **{f"decoder/transformer/layers/[{layer}]/mixer/in_proj/input_block": np.eye(2, dtype=np.float32)},
        )
        return path

    first = write_group("group-0-1.npz", 0)
    second = write_group("group-1-2.npz", 1)
    output = tmp_path / "merged.npz"
    result = run(
        "merge_calibration_groups.py",
        "--output",
        str(output),
        "--layers",
        "2",
        "--group-size",
        "1",
        str(first),
        str(second),
    )
    assert result.returncode == 0, result.stderr
    with np.load(output) as data:
        assert len(data.files) == 2


def test_merge_calibration_groups_rejects_missing_group(tmp_path: Path) -> None:
    first = tmp_path / "group-0-1.npz"
    np.savez(
        first,
        **{"decoder/transformer/layers/[0]/mixer/in_proj/input_block": np.eye(2, dtype=np.float32)},
    )
    output = tmp_path / "merged.npz"
    result = run(
        "merge_calibration_groups.py",
        "--output",
        str(output),
        "--layers",
        "2",
        "--group-size",
        "1",
        str(first),
    )
    assert result.returncode == 1
    assert "missing groups" in result.stderr
