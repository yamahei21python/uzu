from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from env_fingerprint import file_sha256, fingerprint, lalamo_code_hashes, parse_direct_url  # noqa: E402


def test_git_install_records_commit() -> None:
    text = json.dumps(
        {
            "url": "https://github.com/trymirai/lalamo",
            "vcs_info": {"vcs": "git", "commit_id": "e401e17761f2fcd337b0cf20218d67d656e6ecba"},
        }
    )
    parsed = parse_direct_url(text)
    assert parsed["commit"].startswith("e401e177")


def test_pypi_install_has_no_commit() -> None:
    parsed = parse_direct_url(json.dumps({"url": "https://files.pythonhosted.org/lalamo-0.17.0.tar.gz"}))
    assert parsed.get("commit") is None


def test_invalid_metadata_is_empty() -> None:
    assert parse_direct_url("") == {}
    assert parse_direct_url(None) == {}  # type: ignore[arg-type]


def test_file_sha256_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "x.py"
    path.write_bytes(b"a")
    assert file_sha256(path) == file_sha256(path)
    assert file_sha256(tmp_path / "missing.py") is None


def test_code_hashes_cover_the_rht_files(tmp_path: Path) -> None:
    for relative in ("compressed/incoherence.py", "compressed/int.py", "compressed/hybrid.py", "compressed/quant_params.py", "weight_matrix.py"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(relative, encoding="utf-8")
    hashes = lalamo_code_hashes(tmp_path)
    assert hashes["lalamo_hash_complete"] == "true"
    assert sum(key.endswith("_sha256") for key in hashes) == 5
    assert any("incoherence" in key for key in hashes)


def test_missing_rht_file_is_marked_incomplete(tmp_path: Path) -> None:
    (tmp_path / "compressed").mkdir()
    hashes = lalamo_code_hashes(tmp_path)
    assert hashes["lalamo_hash_complete"] == "false"
    assert "incoherence.py" in hashes["lalamo_hash_missing"]


def test_fingerprint_is_json_serialisable() -> None:
    report = fingerprint()
    assert "python_version" in report
    json.dumps(report)
