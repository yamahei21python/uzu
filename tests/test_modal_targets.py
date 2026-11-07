from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from modal_targets import COMMANDS, default_run_name, spec  # noqa: E402


def test_both_sizes_are_defined() -> None:
    assert set(spec("4b")) == {"repo", "spec_name", "layers", "hidden"}
    assert set(spec("9b")) == {"repo", "spec_name", "layers", "hidden"}


def test_4b_and_9b_point_at_distinct_models() -> None:
    assert spec("4b")["repo"] != spec("9b")["repo"]
    assert "4B" in spec("4b")["repo"]
    assert "9B" in spec("9b")["repo"]


def test_layers_match_the_configs() -> None:
    import json

    for size in ("4b", "9b"):
        config = json.loads((ROOT / f"experiments/qwen35-{size}/config.json").read_text(encoding="utf-8"))
        assert spec(size)["layers"] == config["model"]["layers"]
        assert spec(size)["hidden"] == config["model"]["hidden"]
        assert spec(size)["repo"] == config["model"]["repo"]


def test_unknown_size_fails_loudly() -> None:
    with pytest.raises(SystemExit):
        spec("27b")


def test_default_run_name() -> None:
    assert default_run_name("4b") == "huihui-4b"
    assert default_run_name("9b") == "huihui-9b"


def test_commands_cover_the_pipeline() -> None:
    for required in ("import", "rtn", "calibrate", "yaqa", "eval", "qad", "fetch"):
        assert required in COMMANDS
