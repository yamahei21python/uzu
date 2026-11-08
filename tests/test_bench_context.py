"""Context sweep tasks must scale with the requested length."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_context import build_task  # noqa: E402


def test_task_grows_with_length() -> None:
    short = build_task(128, 8)["messages"][0]["content"]
    long = build_task(2048, 8)["messages"][0]["content"]
    assert len(long) > len(short) * 5


def test_generation_limit_is_separate() -> None:
    task = build_task(512, 4)
    assert task["tokens_limit"] == 4
    assert task["number_of_runs"] == 1
