from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qad_finetune import early_stop_decision  # noqa: E402


def test_final_step_does_not_stop() -> None:
    # 2026-11-07: stopping here raised after the last update and skipped the
    # checkpoint save and export, so a finished 10-step run looked like an error.
    assert early_stop_decision(bad_val_evals=2, patience=2, step=10, steps=10) == "ignore"


def test_single_step_run_never_stops() -> None:
    assert early_stop_decision(bad_val_evals=99, patience=2, step=1, steps=1) == "ignore"


def test_non_improvement_before_the_end_stops() -> None:
    assert early_stop_decision(bad_val_evals=2, patience=2, step=5, steps=10) == "stop"


def test_below_patience_continues() -> None:
    assert early_stop_decision(bad_val_evals=1, patience=2, step=5, steps=10) == "continue"


def test_patience_zero_disables_early_stop() -> None:
    assert early_stop_decision(bad_val_evals=99, patience=0, step=5, steps=10) == "continue"
