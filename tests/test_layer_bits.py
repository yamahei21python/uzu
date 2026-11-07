from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rht_int4_convert import load_layer_bits, resolve_bits  # noqa: E402


RULES = [
    {"match": "mlp/down_projection", "bits": 6},
    {"match": "mixer/out_projection", "bits": 6},
    {"match": "layers/[0]/", "bits": 8},
]


def test_first_matching_rule_wins() -> None:
    # order matters: the mlp/down rule comes first, so it wins over layers/[0]
    assert resolve_bits("decoder/.../layers/[0]/mlp/down_projection", RULES, 4) == 6
    assert resolve_bits("decoder/.../layers/[5]/mlp/down_projection", RULES, 4) == 6
    assert resolve_bits("decoder/.../layers/[5]/mlp/up_projection", RULES, 4) == 4


def test_specific_rule_can_win_when_listed_first() -> None:
    rules = [{"match": "layers/[0]/mlp/down_projection", "bits": 8}, *RULES]
    assert resolve_bits("decoder/.../layers/[0]/mlp/down_projection", rules, 4) == 8
    assert resolve_bits("decoder/.../layers/[1]/mlp/down_projection", rules, 4) == 6


def test_default_applies_when_nothing_matches() -> None:
    assert resolve_bits("unknown", RULES, 4) == 4


def test_load_layer_bits_validates_bits(tmp_path: Path) -> None:
    path = tmp_path / "bits.json"
    path.write_text(json.dumps({"default": 4, "rules": [{"match": "x", "bits": 5}]}), encoding="utf-8")
    try:
        load_layer_bits(path, 4)
    except SystemExit as error:
        assert "unsupported bits" in str(error)
    else:
        raise AssertionError("expected SystemExit for bits=5")


def test_load_layer_bits_requires_match(tmp_path: Path) -> None:
    path = tmp_path / "bits.json"
    path.write_text(json.dumps({"rules": [{"bits": 6}]}), encoding="utf-8")
    try:
        load_layer_bits(path, 4)
    except SystemExit as error:
        assert "match" in str(error)
    else:
        raise AssertionError("expected SystemExit for missing match")


def test_load_layer_bits_summary(tmp_path: Path) -> None:
    path = tmp_path / "bits.json"
    path.write_text(json.dumps({"default": 4, "rules": RULES}), encoding="utf-8")
    summary, rules = load_layer_bits(path, 4)
    assert summary["default"] == 4
    assert len(rules) == 3


def test_should_quantize_with_filters() -> None:
    from rht_int4_convert import should_quantize

    assert should_quantize("decoder/.layers/[0]/.mlp/.up_projection", []) is True
    assert should_quantize("decoder/.layers/[0]/.mlp/.up_projection", ["/[0]/"]) is True
    assert should_quantize("decoder/.layers/[7]/.mlp/.up_projection", ["/[0]/"]) is False
    assert should_quantize("decoder/.layers/[7]/.mlp/.up_projection", ["/[0]/", "/[7]/"]) is True
