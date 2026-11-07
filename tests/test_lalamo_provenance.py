from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lane import lalamo_provenance, parse_direct_url  # noqa: E402


def test_git_install_records_commit() -> None:
    text = json.dumps(
        {
            "url": "https://github.com/trymirai/lalamo",
            "vcs_info": {"vcs": "git", "commit_id": "e401e17761f2fcd337b0cf20218d67d656e6ecba"},
        }
    )
    parsed = parse_direct_url(text)
    assert parsed["commit"].startswith("e401e177")
    assert parsed["source"].endswith("lalamo")


def test_pypi_install_has_no_commit() -> None:
    parsed = parse_direct_url(json.dumps({"url": "https://files.pythonhosted.org/lalamo-0.17.0.tar.gz"}))
    assert parsed.get("commit") is None
    assert "pythonhosted" in parsed["source"]


def test_invalid_metadata_is_empty() -> None:
    assert parse_direct_url("") == {}
    assert parse_direct_url(None) == {}  # type: ignore[arg-type]


def test_installed_lalamo_reports_a_version() -> None:
    seen = lalamo_provenance()
    assert seen.get("lalamo_version")
    assert seen.get("lalamo_source")
