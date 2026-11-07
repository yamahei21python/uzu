from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_lalamo_eval_copy import lalamo_config  # noqa: E402


def write_model(tmp_path: Path, config: dict) -> Path:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return model


def test_normalized_config_is_reverted_for_lalamo(tmp_path: Path) -> None:
    """The uzu import drops response_format and adds two keys lalamo rejects.

    Pinned lalamo e401e177 declares ``response_format`` without a default and
    rejects extra fields, so calibration fails on the imported directory.
    """
    normalized = {
        "token_codec_config": {
            "end_of_thinking_tag": "</think>",
            "output_parser_regex": ".*",
            "eos_token": "<|endoftext|>",
        }
    }
    restored = lalamo_config(write_model(tmp_path, normalized))
    token_codec = restored["token_codec_config"]
    assert "end_of_thinking_tag" not in token_codec
    assert "output_parser_regex" not in token_codec
    assert "response_format" in token_codec
    assert token_codec["response_format"]["prefixes"] == [" think", " response"]


def test_saved_lalamo_config_wins(tmp_path: Path) -> None:
    model = write_model(tmp_path, {"token_codec_config": {}})
    (model / "config.lalamo.json").write_text(json.dumps({"already": "lalamo"}), encoding="utf-8")
    assert lalamo_config(model) == {"already": "lalamo"}


def test_config_without_token_codec_is_untouched(tmp_path: Path) -> None:
    original = {"layers": 32}
    assert lalamo_config(write_model(tmp_path, original)) == original
