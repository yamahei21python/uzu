from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANES = ROOT / "lanes"


def run_lane(spec: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_lane.py"), "--spec", str(spec), "--dry-run", *extra],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_clean_suite_spec_builds_guarded_commands() -> None:
    result = run_lane(LANES / "clean-suite-08b.json")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("$ ")]
    calibration = [line for line in commands if "yaqa_calibration.py" in line]
    assert len(calibration) == 3
    for line in calibration:
        # both eval corpora must be passed so the overlap guard runs
        assert line.count("--eval-corpus") == 2
        assert "--raw-output" not in line
    imports = [line for line in commands if "rht_int4_convert.py" in line and "--dry-run" in line]
    conversions = [line for line in commands if "rht_int4_convert.py" in line and "--dry-run" not in line]
    assert len(imports) == 1
    assert len(conversions) == 3
    # the import output is uzu-normalized; calibration needs the lalamo copy
    assert len([line for line in commands if "make_lalamo_eval_copy.py" in line]) == 4
    assert len([line for line in commands if "eval_quantized.py" in line]) == 6


def test_balanced_spec_skips_raw_conversion_and_merges_three_ratios() -> None:
    result = run_lane(LANES / "balanced-h-08b.json")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("$ ")]
    raw = [line for line in commands if "yaqa_calibration.py" in line]
    assert len(raw) == 2
    assert all("--raw-output" in line for line in raw)
    merge = [line for line in commands if "merge_language_hessians.py" in line]
    assert len(merge) == 3
    ratios = [line.split("--ja-ratio ")[1].split()[0] for line in merge]
    assert ratios == ["0.5", "0.65", "0.35"]
    convert = [
        line for line in commands if "rht_int4_convert.py" in line and "--dry-run" not in line
    ]
    # raw conditions are not converted, only the three merged ones
    assert len(convert) == 3


def test_missing_spec_fields_fail_loudly(tmp_path: Path) -> None:
    spec = tmp_path / "bad.json"
    spec.write_text(json.dumps({"lane": "bad", "paths": {"kit": "kit", "work": str(tmp_path)}}), encoding="utf-8")
    result = run_lane(spec)
    assert result.returncode != 0
    assert "no conditions" in result.stderr


def test_resume_skips_completed_condition(tmp_path: Path) -> None:
    spec = json.loads((LANES / "clean-suite-08b.json").read_text(encoding="utf-8"))
    spec["paths"]["work"] = str(tmp_path)
    done = tmp_path / "out" / spec["lane"] / "mix65"
    done.mkdir(parents=True)
    (done / "result.json").write_text(json.dumps({"name": "mix65", "status": "complete"}), encoding="utf-8")
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    result = run_lane(spec_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "skip mix65" in result.stdout
    calibration = [line for line in result.stdout.splitlines() if "yaqa_calibration.py" in line]
    assert len(calibration) == 2


def test_validation_flags_chunk_drift() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from run_lane import Runner
    finally:
        sys.path.pop(0)

    runner = Runner(
        {"lane": "t", "eval": {"max_sequences": 100}},
        ROOT / "lanes" / "clean-suite-08b.json",
        dry_run=True,
    )
    problems = runner.validate(
        [
            {"name": "a", "ppl": {"en": {"chunks": 100, "bf16_ppl": 35.38}}},
            {"name": "b", "ppl": {"en": {"chunks": 20, "bf16_ppl": 35.38}}},
        ]
    )
    assert any("b/en: chunks=20" in problem for problem in problems)


def test_validation_flags_baseline_drift() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from run_lane import Runner
    finally:
        sys.path.pop(0)

    runner = Runner(
        {"lane": "t", "eval": {"max_sequences": 100}},
        ROOT / "lanes" / "clean-suite-08b.json",
        dry_run=True,
    )
    problems = runner.validate(
        [
            {"name": "a", "ppl": {"en": {"chunks": 100, "bf16_ppl": 35.38}}},
            {"name": "b", "ppl": {"en": {"chunks": 100, "bf16_ppl": 39.10}}},
        ]
    )
    assert any("bf16_ppl drift" in problem for problem in problems)
