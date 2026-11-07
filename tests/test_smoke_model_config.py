from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_model_config import checks, hf_geometry  # noqa: E402


def test_hf_geometry_reads_text_config() -> None:
    config = {"text_config": {"hidden_size": 4096, "num_hidden_layers": 32, "vocab_size": 248320}}
    assert hf_geometry(config) == {"hidden": 4096, "layers": 32, "vocab": 248320}


def test_matching_geometry_is_clean() -> None:
    local = {"hidden": 2560, "layers": 32, "tied_embedding": True}
    remote = {"hidden": 2560, "layers": 32, "vocab": 248320}
    assert checks({"tied_embedding": True}, local, remote) == []


def test_hidden_must_divide_the_block_size() -> None:
    local = {"hidden": 2559, "layers": 32, "tied_embedding": True}
    problems = checks({"tied_embedding": True}, local, {"hidden": 2559, "layers": 32, "vocab": 248320})
    assert any(problem["check"] == "hidden-divisible" for problem in problems)


def test_layer_drift_is_reported() -> None:
    local = {"hidden": 2560, "layers": 31, "tied_embedding": True}
    problems = checks({"tied_embedding": True}, local, {"hidden": 2560, "layers": 32, "vocab": 248320})
    assert any(problem["check"] == "layers-match-hf" for problem in problems)


def test_offline_is_a_problem_not_a_crash() -> None:
    problems = checks({"tied_embedding": True}, {"hidden": 1024, "layers": 24, "tied_embedding": True}, {})
    assert any(problem["check"] == "hf-config" for problem in problems)
