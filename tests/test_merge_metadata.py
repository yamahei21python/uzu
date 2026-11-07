from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def write_raw(tmp_path: Path, name: str, corpus: str) -> tuple[Path, Path]:
    key = "decoder/transformer/layers/[0]/mixer/in_proj/input_block"
    matrix = np.eye(4, dtype=np.float32) * 2.0
    npz = tmp_path / f"{name}.npz"
    np.savez(npz, **{key: matrix})
    payload = {
        "model_dir": "bf16-lalamo",
        "corpus": corpus,
        "samples": 32,
        "max_tokens": 128,
        "output_block": "small",
        "output_block_limit": 2048,
        "damping": 0.001,
        "raw_output": True,
        "overlap_checked": True,
    }
    json_path = tmp_path / f"{name}.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return npz, json_path


def run_merge(tmp_path: Path, en_json: Path | None, ja_json: Path | None) -> subprocess.CompletedProcess[str]:
    en_npz, _ = write_raw(tmp_path, "en", "eval_en_dedup.txt")
    ja_npz, _ = write_raw(tmp_path, "ja", "eval_ja_dedup.txt")
    command = [
        sys.executable,
        str(ROOT / "scripts" / "merge_language_hessians.py"),
        "--en",
        str(en_npz),
        "--ja",
        str(ja_npz),
        "--ja-ratio",
        "0.5",
        "--damping",
        "1e-3",
        "--output",
        str(tmp_path / "merged.npz"),
        "--json-out",
        str(tmp_path / "merged.json"),
    ]
    if en_json:
        command += ["--en-json", str(en_json)]
    if ja_json:
        command += ["--ja-json", str(ja_json)]
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)


def test_merge_carries_calibration_metadata(tmp_path: Path) -> None:
    en_npz, _ = write_raw(tmp_path, "en", "calibration_en_dedup.txt")
    ja_npz, _ = write_raw(tmp_path, "ja", "calibration_ja_dedup.txt")
    _, en_json = write_raw(tmp_path, "en2", "calibration_en_dedup.txt")
    _, ja_json = write_raw(tmp_path, "ja2", "calibration_ja_dedup.txt")
    result = run_merge(tmp_path, en_json, ja_json)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))
    assert summary["samples"] == 32
    assert summary["max_tokens"] == 128
    assert summary["output_block"] == "small"
    assert summary["damping"] == 0.001
    assert summary["ja_ratio"] == 0.5
    assert set(summary["corpus"]) == {"calibration_en_dedup.txt", "calibration_ja_dedup.txt"}
    assert summary["sources"]["en"]["raw_output"] is True


def test_merge_without_jsons_still_works(tmp_path: Path) -> None:
    result = run_merge(tmp_path, None, None)
    assert result.returncode == 0, result.stderr
    summary = json.loads((tmp_path / "merged.json").read_text(encoding="utf-8"))
    assert "samples" not in summary
    assert summary["sources"] == {}
