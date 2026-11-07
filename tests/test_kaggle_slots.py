from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_slots import count_running, is_running  # noqa: E402


def test_running_detection() -> None:
    assert is_running('has status "KernelWorkerStatus.RUNNING"')
    assert is_running("QUEUED")
    assert not is_running('has status "KernelWorkerStatus.COMPLETE"')
    assert not is_running("KernelWorkerStatus.ERROR")


def test_count_running() -> None:
    assert count_running(["RUNNING", "COMPLETE", "running"]) == 2
    assert count_running([]) == 0
