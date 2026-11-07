from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from publish_kit import (  # noqa: E402
    parse_dataset_listing,
    parse_dataset_listing_json,
    parse_manifest_commit,
    verify_listing,
)


def test_listing_parses_paths() -> None:
    text = (
        "name                                            size  creationDate\n"
        "--------------------------------------------  -------  --------------------------\n"
        "kit_manifest.json                                6669  2026-11-07 07:58:21\n"
        "scripts/preflight.py                             1234  2026-11-07 07:59:02\n"
        "data/eval_en_wikitext2_dedup.txt               502257  2026-11-07 07:59:02\n"
        "Next Page Token = CfDJ8FwkkMkekkBCtzoFDl4Vj7g\n"
    )
    names = parse_dataset_listing(text)
    assert "scripts/preflight.py" in names
    assert "data/eval_en_wikitext2_dedup.txt" in names
    assert "name" not in names


def test_gutted_dataset_is_rejected() -> None:
    problems = verify_listing(["kit_manifest.json"])
    assert len(problems) == 3
    assert any("scripts/" in problem for problem in problems)


def test_complete_listing_passes() -> None:
    names = ["kit_manifest.json", "scripts/x.py", "data/x.txt", "configs/x.json"]
    assert verify_listing(names) == []


def test_manifest_commit_parsing() -> None:
    assert parse_manifest_commit('{"built_from_commit": "abc"}') == "abc"
    assert parse_manifest_commit("{}") is None
    assert parse_manifest_commit("not json") is None


def test_json_listing_parses_after_token_line() -> None:
    text = 'Next Page Token = abc\n[\n  {"name": "scripts/a.py", "size": 1}\n]\n'
    assert parse_dataset_listing_json(text) == ["scripts/a.py"]
