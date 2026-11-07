from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ag_queue import is_pressure, parse_created, select_by_size  # noqa: E402


def env(env_id: str, size_mib: int, created: str) -> dict:
    return {"id": env_id, "size_bytes": str(size_mib * 1024 * 1024), "created": created}


def test_parse_created_returns_utc_naive() -> None:
    # 03:00Z must not be shifted into local time, otherwise age calculations go negative
    parsed = parse_created("2026-11-07T03:00:00Z")
    assert parsed == datetime(2026, 11, 7, 3, 0, 0)


def test_is_pressure_covers_storage_quota_message() -> None:
    message = '{"error":{"message":"Project environment storage quota exceeded.","code":"too_many_requests"}}'
    assert is_pressure(message)
    assert is_pressure("Rate limit exceeded")
    assert is_pressure("RESOURCE_EXHAUSTED")
    assert not is_pressure("403 forbidden")


def test_select_by_size_takes_the_largest_first() -> None:
    environments = [
        env("big", 1000, "2026-11-07T00:00:00Z"),
        env("medium", 200, "2026-11-07T00:10:00Z"),
        env("tiny", 10, "2026-11-07T00:20:00Z"),
    ]
    selected = select_by_size(
        environments,
        keep=set(),
        target_gib=0.5 * 1024 / 1024,  # 0.5 GiB
        min_age_minutes=0,
        now=datetime(2026, 11, 7, 12, 0, 0),
    )
    assert [item["id"] for item in selected] == ["big"]
    # 1000 MiB removed leaves 210 MiB, already under the 512 MiB target


def test_select_by_size_skips_protected_and_young() -> None:
    environments = [
        env("protected", 900, "2026-11-07T00:00:00Z"),
        env("young", 900, "2026-11-07T11:55:00Z"),
        env("old", 300, "2026-11-07T00:00:00Z"),
    ]
    selected = select_by_size(
        environments,
        keep={"protected"},
        target_gib=1.0,
        min_age_minutes=10,
        now=datetime(2026, 11, 7, 12, 0, 0),
    )
    assert [item["id"] for item in selected] == ["old"]


def test_select_by_size_stops_at_target() -> None:
    environments = [env("a", 100, "2026-11-07T00:00:00Z")]
    selected = select_by_size(
        environments,
        keep=set(),
        target_gib=1.0,
        min_age_minutes=0,
        now=datetime(2026, 11, 7, 12, 0, 0),
    )
    assert selected == []


def test_cleanup_offline_auto_mode(tmp_path: Path) -> None:
    env_list = tmp_path / "envs.json"
    env_list.write_text(
        json.dumps(
            {
                "environments": [
                    env("big", 1000, "2026-11-07T00:00:00Z"),
                    env("small", 20, "2026-11-07T00:10:00Z"),
                ]
            }
        ),
        encoding="utf-8",
    )
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "ag_queue.py"),
            "--root",
            str(tmp_path),
            "cleanup-envs",
            "--mode",
            "auto",
            "--target-gib",
            "0.5",
            "--min-age-minutes",
            "0",
            "--dry-run",
            "--env-list-file",
            str(env_list),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "delete big" in result.stdout
