from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight import parse_capabilities, verdict  # noqa: E402


def test_parse_capabilities() -> None:
    output = "7.5, Tesla T4\n8.0, A100-SXM4-40GB\nbad\n"
    assert parse_capabilities(output) == [7.5, 8.0]


def test_t4_verdict_uses_float32_and_xla() -> None:
    result = verdict([7.5], None)
    assert result["ampere_or_newer"] is False
    assert result["bf16_matmul"] is False
    assert result["eval_dtype"] == "float32"
    assert result["attention"] == "xla"


def test_ampere_verdict_keeps_defaults() -> None:
    result = verdict([8.0], None)
    assert result["ampere_or_newer"] is True
    assert result["bf16_matmul"] is True
    assert result["eval_dtype"] == "bfloat16"
    assert result["attention"] == "default"


def test_measurement_overrides_capability() -> None:
    # hardware says Ampere but the measured dot failed: trust the measurement
    result = verdict([8.0], False)
    assert result["bf16_matmul"] is False
    assert result["bf16_source"] == "measured"


def test_sdpa_failure_marks_attention_unavailable() -> None:
    result = verdict([7.5], False, {"dot": True, "dot_f32_acc": False, "sdpa_xla": False})
    assert result["attention"] == "unavailable"
    assert "CPU or Modal" in result["note"]


def test_all_probes_pass_keeps_default_attention() -> None:
    result = verdict([8.0], True, {"dot": True, "dot_f32_acc": True, "sdpa_xla": True})
    assert result["attention"] == "default"
    assert result["eval_dtype"] == "bfloat16"


def test_no_gpu_is_not_ampere() -> None:
    result = verdict([], None)
    assert result["ampere_or_newer"] is False
    assert result["bf16_matmul"] is False
