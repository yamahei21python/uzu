from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import ag_queue  # noqa: E402


def test_env_total_gib_without_key(monkeypatch) -> None:
    monkeypatch.setattr(ag_queue, "api_key", lambda: None)
    assert ag_queue.env_total_gib() is None


def test_cleanup_under_pressure_escalates_three_stages() -> None:
    calls = []

    def fake_cleanup(**kwargs):
        calls.append(kwargs)
        # first two stages free nothing, third succeeds
        after = 1.2 if len(calls) < 3 else 0.4
        return {"total_gib_after": after, "deleted": 1}

    original = ag_queue.cleanup_envs
    ag_queue.cleanup_envs = fake_cleanup
    try:
        summary = ag_queue.cleanup_under_pressure(target_gib=0.8)
    finally:
        ag_queue.cleanup_envs = original

    assert summary["total_gib_after"] == 0.4
    assert len(calls) == 3
    assert calls[0]["protect_sessions"] is True
    assert calls[1]["protect_sessions"] is False
    assert calls[2]["force"] is True
