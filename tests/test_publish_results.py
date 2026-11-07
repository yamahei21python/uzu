from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from publish_results import collect_issues, result_files  # noqa: E402


def write_manifest(root: Path, lane: str, payload: dict) -> None:
    lane_dir = root / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_missing_manifest_is_an_error(tmp_path: Path) -> None:
    errors, warnings = collect_issues(tmp_path)
    assert errors and "no manifest.json" in errors[0]
    assert warnings == []


def test_clean_manifest_passes(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        "clean-suite-08b",
        {
            "lane": "clean-suite-08b",
            "problems": [],
            "environment_fingerprint": {"lalamo_commit": "e401e177", "lalamo_hash_complete": "true"},
        },
    )
    errors, warnings = collect_issues(tmp_path)
    assert errors == []
    assert warnings == []


def test_reported_problems_block_publish(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        "smoke-08b",
        {"lane": "smoke-08b", "problems": ["bf16_ppl drift"], "environment_fingerprint": {"lalamo_hash_complete": "true"}},
    )
    errors, _ = collect_issues(tmp_path)
    assert len(errors) == 1
    assert "bf16_ppl drift" in errors[0]


def test_missing_fingerprint_warns_but_publishes(tmp_path: Path) -> None:
    write_manifest(tmp_path, "old-lane", {"lane": "old-lane", "problems": []})
    errors, warnings = collect_issues(tmp_path)
    assert errors == []
    assert any("environment_fingerprint" in warning for warning in warnings)


def test_incomplete_hash_warns_about_comparison(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        "lane",
        {"lane": "lane", "problems": [], "environment_fingerprint": {"lalamo_hash_complete": "false"}},
    )
    _, warnings = collect_issues(tmp_path)
    assert any("do not compare" in warning for warning in warnings)


def test_result_files_skip_caches(tmp_path: Path) -> None:
    (tmp_path / "lane").mkdir()
    (tmp_path / "lane" / "result.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "x.pyc").write_bytes(b"x")
    files = result_files(tmp_path)
    assert "lane/result.json" in files
    assert not any("__pycache__" in name for name in files)
