"""Bench results must carry their load conditions and use medians."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bench_protocol import summarize  # noqa: E402


def test_summarize_uses_medians_and_keeps_loads() -> None:
    trials = [
        {"load": 1.2, "generate": [100.0, 102.0], "prompt": [500.0], "ttft": [0.02], "memory_bytes": 10},
        {"load": 1.4, "generate": [98.0, 96.0], "prompt": [480.0], "ttft": [0.03], "memory_bytes": 10},
    ]
    summary = summarize(trials)
    assert summary["generate_median"] == 99.0
    assert summary["generate_min"] == 96.0
    assert summary["generate_max"] == 102.0
    assert summary["prompt_median"] == 490.0
    assert summary["loads"] == [1.2, 1.4]
    assert summary["trials"] == 2
