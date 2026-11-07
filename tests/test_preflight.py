from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preflight import parse_capabilities, verdict  # noqa: E402

ALL_PASS = {
    "dot": "pass",
    "dot_f32_acc": "pass",
    "sdpa_xla": "pass",
    "quant_roundtrip": "pass",
    "qad_grad": "pass",
    "tokenizer": "pass",
}


def test_parse_capabilities() -> None:
    output = "7.5, Tesla T4\n8.0, A100-SXM4-40GB\nbad\n"
    assert parse_capabilities(output) == [7.5, 8.0]


def test_t4_profile() -> None:
    probes = dict(ALL_PASS, dot_f32_acc="fail", sdpa_xla="fail")
    result = verdict([7.5], probes)
    assert result["ampere_or_newer"] is False
    assert result["bf16_matmul"] is False
    assert result["attention"] == "unavailable"
    assert result["eval_dtype"] == "float32"
    assert result["lane_ok"] is False
    assert "CPU or Modal" in result["note"]


def test_ampere_profile() -> None:
    result = verdict([8.0], ALL_PASS)
    assert result["ampere_or_newer"] is True
    assert result["bf16_matmul"] is True
    assert result["attention"] == "default"
    assert result["eval_dtype"] == "bfloat16"
    assert result["lane_ok"] is True


def test_quant_probe_failure_blocks_the_lane() -> None:
    probes = dict(ALL_PASS, quant_roundtrip="fail")
    result = verdict([8.0], probes)
    assert result["lane_ok"] is False
    assert "quant_roundtrip" in result["probe_failures"]


def test_skipped_probe_does_not_block() -> None:
    probes = dict(ALL_PASS, quant_roundtrip="skip", tokenizer="skip")
    assert verdict([8.0], probes)["lane_ok"] is True


def test_no_gpu_with_skips() -> None:
    result = verdict([], {})
    assert result["ampere_or_newer"] is False
    assert result["bf16_matmul"] is False


def test_qad_probe_failure_does_not_block_ordinary_lanes() -> None:
    probes = dict(ALL_PASS, qad_grad="fail")
    result = verdict([8.0], probes)
    assert result["lane_ok"] is True
    assert result["qad_ok"] is False
    assert "qad_grad" not in result["probe_failures"]


def test_missing_lalamo_submodule_is_a_failure() -> None:
    from preflight import classify_probe_failure

    message = "ModuleNotFoundError: No module named 'lalamo.compressed.incoherence'"
    assert classify_probe_failure("quant_roundtrip", message)[0] == "fail"


def test_missing_lalamo_package_is_a_skip() -> None:
    from preflight import classify_probe_failure

    message = "ModuleNotFoundError: No module named 'lalamo'"
    assert classify_probe_failure("quant_roundtrip", message)[0] == "skip"
