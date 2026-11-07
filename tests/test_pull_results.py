from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_manifest(root: Path, lane: str, conditions: list[str], problems: list[str] | None = None) -> None:
    directory = root / lane
    directory.mkdir(parents=True)
    payload = {
        "lane": lane,
        "results": [{"name": name, "status": "complete"} for name in conditions],
        "problems": problems or [],
        "spec": f"lanes/{lane}.json",
        "updated": "2026-11-07T14:00:00",
    }
    (directory / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def run(root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "pull_results.py"),
            "--out",
            str(root),
            "--skip-download",
            *extra,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_summarises_lanes_and_conditions(tmp_path: Path) -> None:
    write_manifest(tmp_path, "clean-suite-08b", ["mix65", "mix35", "jaonly"])
    write_manifest(tmp_path, "balanced-h-08b", ["balanced-50", "balanced-65"])
    out = tmp_path / "summary.json"
    result = run(tmp_path, "--json", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["lane_count"] == 2
    assert summary["condition_count"] == 5
    assert summary["problems"] == []


def test_reports_manifest_problems_and_can_fail(tmp_path: Path) -> None:
    write_manifest(tmp_path, "clean-suite-08b", ["mix65"], problems=["mix65/en: chunks=20"])
    result = run(tmp_path, "--strict")
    assert result.returncode == 1
    assert "chunks=20" in result.stdout


def test_missing_root_fails(tmp_path: Path) -> None:
    result = run(tmp_path / "absent")
    assert result.returncode != 0
    assert "nothing at" in result.stderr
