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


def test_identical_val_is_not_a_bad_eval() -> None:
    # 2026-11-07 uzu-qad-cpu: val_kl was 0.262012 at step 0 and at step 5, the
    # second eval counted as a non-improvement, and early stop failed the kernel.
    # An identical value means the weights did not move, not that we regressed.
    from qad_finetune import validation_is_unchanged

    assert validation_is_unchanged(0.262012, 0.262012) is True
    assert validation_is_unchanged(0.262012, 0.262013) is False
    assert validation_is_unchanged(None, 0.5) is False


def test_bf16_master_weights_silently_drop_a_finetuning_lr_update() -> None:
    import pytest

    jnp = pytest.importorskip("jax.numpy")

    # master_weights are around 0.008 (measured on the 0.8B student); one bf16
    # ulp there is 6.1e-5, so a 2e-5 AdamW step rounds back to the same value.
    # It never accumulates: every step rounds from the stored bf16 value, which
    # is why val_kl stayed bit-identical for the whole uzu-qad-cpu run.
    weight = jnp.array([0.008], dtype=jnp.bfloat16)
    moved = weight
    for _ in range(100):
        moved = moved + jnp.array([2e-5], dtype=jnp.bfloat16)
    assert moved == weight


def test_float32_master_weights_keep_the_same_update() -> None:
    import pytest

    jnp = pytest.importorskip("jax.numpy")

    weight = jnp.array([0.008], dtype=jnp.float32)
    moved = weight + jnp.array([2e-5], dtype=jnp.float32)
    assert float(moved[0] - weight[0]) > 0.0
    assert float(moved[0]) != float(weight[0])


def test_median_quantization_step_uses_the_stored_scales() -> None:
    import numpy as np

    from qad_finetune import median_quantization_step

    assert median_quantization_step([]) is None
    # scales are per-group; 4-bit splits the group range into 7 steps
    assert abs(median_quantization_step([np.array([0.007, 0.007])]) - 0.007 / 7) < 1e-9


def test_finetuning_lr_is_flagged_as_absorbed_by_the_quantizer() -> None:
    from qad_finetune import update_is_absorbed

    # measured on the 0.8B student: median scale 0.0033 -> int4 step 4.7e-4
    step = 0.00047
    assert update_is_absorbed(2e-5, step) is True
    assert update_is_absorbed(1e-3, step) is False
    assert update_is_absorbed(1e-3, None) is False


def test_peak_rss_is_reported_in_mb() -> None:
    from qad_finetune import peak_rss_mb

    value = peak_rss_mb()
    assert value > 0
    assert value < 200_000  # sanity: not bytes


def test_checkpoint_params_only_flag_is_registered() -> None:
    import subprocess

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "qad_finetune.py"), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert "--checkpoint-params-only" in result.stdout
    assert "AdamW moments" in result.stdout


def test_stale_lalamo_backup_is_detected(tmp_path) -> None:
    import os
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    from make_lalamo_eval_copy import fresh_backup

    weights = tmp_path / "model.safetensors"
    backup = tmp_path / "model.safetensors.lalamo.bak"
    weights.write_bytes(b"new")
    backup.write_bytes(b"old")
    os.utime(backup, (1_000_000, 1_000_000))
    os.utime(weights, (2_000_000, 2_000_000))
    assert fresh_backup(weights, backup) is False
    os.utime(backup, (3_000_000, 3_000_000))
    assert fresh_backup(weights, backup) is True
