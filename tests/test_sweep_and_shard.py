from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_lane(spec: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_lane.py"), "--spec", str(spec), "--dry-run", *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def generate(out_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "make_sweep_spec.py"),
            "--base",
            str(ROOT / "lanes" / "clean-suite-08b.json"),
            "--out-dir",
            str(out_dir),
            "--name",
            "robustness-08b",
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_sweep_generates_three_shards_with_unique_conditions(tmp_path: Path) -> None:
    result = generate(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    specs = sorted(tmp_path.glob("robustness-08b-shard*.json"))
    assert len(specs) == 3
    names: list[str] = []
    for path in specs:
        spec = json.loads(path.read_text())
        assert spec["lane"] == path.stem
        assert "merge" not in spec
        assert len(spec["conditions"]) == 9
        names.extend(condition["name"] for condition in spec["conditions"])
    assert len(names) == len(set(names)) == 27


def test_sweep_keeps_eval_tokens_in_step_with_calibration(tmp_path: Path) -> None:
    generate(tmp_path)
    for path in tmp_path.glob("robustness-08b-shard*.json"):
        spec = json.loads(path.read_text())
        seqs = {condition["max_tokens"] for condition in spec["conditions"]}
        # each shard spreads over the grid, so eval max_tokens must stay a sane default
        assert spec["eval"]["max_sequences"] == 100
        assert seqs <= {128, 256, 512}


def test_shard_flag_slices_conditions_and_renames_lane() -> None:
    result = run_lane(ROOT / "lanes" / "clean-suite-08b.json", "--shard", "0/2")
    assert result.returncode == 0, result.stdout + result.stderr
    calibration = [line for line in result.stdout.splitlines() if line.startswith("$ ") and "yaqa_calibration.py" in line]
    assert len(calibration) == 2
    assert "clean-suite-08b-shard1of2" in result.stdout


def test_shard_rejects_merge_specs() -> None:
    result = run_lane(ROOT / "lanes" / "balanced-h-08b.json", "--shard", "0/2")
    assert result.returncode != 0
    assert "--shard cannot be combined" in result.stderr


def test_shard_out_of_range_fails() -> None:
    result = run_lane(ROOT / "lanes" / "clean-suite-08b.json", "--shard", "5/3")
    assert result.returncode != 0
    assert "out of range" in result.stderr
