from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_usage import is_terminal, read_entries, summarize, week_key  # noqa: E402


def test_week_key_is_iso() -> None:
    assert week_key(date(2026, 11, 7)) == "2026-W45"


def test_terminal_statuses() -> None:
    assert is_terminal('kernel has status "KernelWorkerStatus.COMPLETE"')
    assert is_terminal("KernelWorkerStatus.ERROR")
    assert is_terminal("cancelAcknowledged")
    assert not is_terminal("KernelWorkerStatus.RUNNING")


def test_summarize_groups_by_account_and_respects_week() -> None:
    entries = [
        {"week": "2026-W45", "account": "owner", "seconds": 3600, "kernel": "a"},
        {"week": "2026-W45", "account": "owner", "seconds": 1800, "kernel": "b"},
        {"week": "2026-W45", "account": "muse", "seconds": 7200, "kernel": "c"},
        {"week": "2026-W44", "account": "owner", "seconds": 9999, "kernel": "old"},
    ]
    summary = summarize(entries, "2026-W45", quota_hours=30)
    assert summary["runs"] == 3
    assert summary["accounts"]["owner"]["hours"] == 1.5
    assert summary["accounts"]["owner"]["remaining_hours"] == 28.5
    assert summary["accounts"]["muse"]["hours"] == 2.0
    assert summary["accounts"]["owner"]["used_pct"] == 5.0


def test_cli_log_and_report(tmp_path: Path) -> None:
    ledger = tmp_path / "usage.jsonl"
    week = week_key(date.today())
    log = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_usage.py"),
            "--ledger",
            str(ledger),
            "log",
            "--kernel",
            "owner/k",
            "--seconds",
            "5400",
            "--note",
            "test",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert log.returncode == 0, log.stderr
    entries = read_entries(ledger)
    assert len(entries) == 1

    report = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "kaggle_usage.py"),
            "--ledger",
            str(ledger),
            "report",
            "--week",
            week,
            "--quota-hours",
            "30",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert report.returncode == 0, report.stderr
    assert "1.50" in report.stdout
    assert "owner" in report.stdout


def test_corrupt_lines_are_skipped(tmp_path: Path) -> None:
    ledger = tmp_path / "usage.jsonl"
    ledger.write_text('{"week": "2026-W45", "account": "owner", "seconds": 60}\nnot json\n', encoding="utf-8")
    entries = read_entries(ledger)
    assert len(entries) == 1
