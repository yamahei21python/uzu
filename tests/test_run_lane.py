from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANES = ROOT / "lanes"
sys.path.insert(0, str(ROOT / "scripts"))

from run_lane import run_conditions  # noqa: E402


class FakeRunner:
    def __init__(self) -> None:
        self.ran: list[str] = []

    def run_condition(self, condition: dict, bf16: Path | None) -> dict:
        name = condition["name"]
        self.ran.append(name)
        if name == "bad":
            raise SystemExit("calibration failed with code -9")
        return {"name": name, "status": "complete"}


def test_failed_condition_does_not_stop_the_lane() -> None:
    # a single OOM used to abort the lane and lose every later condition
    runner = FakeRunner()
    summaries, failures = run_conditions(runner, [{"name": "a"}, {"name": "bad"}, {"name": "c"}], None)
    assert runner.ran == ["a", "bad", "c"]
    assert [summary["status"] for summary in summaries] == ["complete", "failed", "complete"]
    assert failures and failures[0].startswith("bad:")


def test_results_are_flushed_after_every_condition() -> None:
    uploads: list[int] = []
    run_conditions(FakeRunner(), [{"name": "a"}, {"name": "b"}], None, after_each=lambda: uploads.append(1))
    assert uploads == [1, 1]


def test_fail_fast_still_available() -> None:
    with pytest.raises(SystemExit):
        run_conditions(FakeRunner(), [{"name": "bad"}, {"name": "c"}], None, fail_fast=True)


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


def test_smoke_spec_is_cheap_and_single_condition() -> None:
    result = run_lane(LANES / "smoke-08b.json")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("$ ")]
    calibration = [line for line in commands if "yaqa_calibration.py" in line]
    assert len(calibration) == 1
    assert "--samples 2" in calibration[0]
    assert "--max-tokens 32" in calibration[0]
    evals = [line for line in commands if "eval_quantized.py" in line]
    assert len(evals) == 2
    assert all("--max-sequences 2" in line for line in evals)


def test_calib_dtype_is_passed_to_calibration() -> None:
    result = run_lane(LANES / "damping-ablation-08b.json")
    assert result.returncode == 0, result.stdout + result.stderr
    calibration = [
        line for line in result.stdout.splitlines() if line.startswith("$ ") and "yaqa_calibration.py" in line
    ]
    assert len(calibration) == 2
    assert all("--calib-dtype float32" in line for line in calibration)
    assert all("--damping 0.01" in line for line in calibration)


def test_rtn_condition_skips_calibration_and_preconditioners() -> None:
    result = run_lane(LANES / "eval-4b-rtn.json")
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("$ ")]
    assert not [line for line in commands if "yaqa_calibration.py" in line]
    conversions = [
        line for line in commands if "rht_int4_convert.py" in line and "--load-from" in line
    ]
    assert len(conversions) == 1
    assert "--preconditioners" not in conversions[0]
    assert len([line for line in commands if "eval_quantized.py" in line]) == 2


def test_seed_and_convert_overrides_reach_the_commands() -> None:
    seeds = run_lane(LANES / "seed-variance-08b.json")
    assert "--seed 0" in seeds.stdout and "--seed 1" in seeds.stdout and "--seed 2" in seeds.stdout

    groups = run_lane(LANES / "group-size-ablation-08b.json")
    for size in ("32", "64", "128"):
        assert f"--group-size {size}" in groups.stdout

    bits = run_lane(LANES / "bit-alloc-ablation-08b.json")
    assert "--layer-bits-json" in bits.stdout
    assert bits.stdout.count("--layer-bits-json") == 1


def test_cli_overrides_replace_spec_settings() -> None:
    result = run_lane(
        LANES / "seed-variance-08b.json",
        "--jax-platform", "cpu",
        "--eval-dtype", "float32",
        "--parallel", "replicated",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("$ ")]
    calibration = [line for line in commands if "yaqa_calibration.py" in line]
    evals = [line for line in commands if "eval_quantized.py" in line]
    assert all("--parallel replicated" in line for line in calibration)
    assert all("--dtype float32" in line for line in evals)


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
