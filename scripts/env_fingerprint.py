#!/usr/bin/env python3
"""Canonical environment fingerprint for quantization runs.

Version strings lie. PyPI ``lalamo==0.17.0`` (2026-09-24) and the pinned git
commit ``e401e177`` (2026-11-02) both report 0.17.0, but the PyPI tree has no
``compressed/incoherence.py``, so RHT cannot run there. Comparing results from
those two installs is meaningless even though every version field matches.

The fingerprint records two independent identities:

1. where the package came from (``direct_url.json``: VCS URL + commit)
2. hashes of the files that change numerics (RHT / INT4 / weight layout)

(2) is the one to trust: identical hashes mean identical quantize/dequantize
code, regardless of how the package was installed.

Usage:
    python scripts/env_fingerprint.py
    python scripts/env_fingerprint.py --json out.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import distribution, version
from pathlib import Path

# Files that directly determine the quantized weights. A change in any of
# these can move BPC, so they belong in every result manifest.
FINGERPRINT_FILES = (
    "compressed/incoherence.py",
    "compressed/int.py",
    "compressed/hybrid.py",
    "compressed/quant_params.py",
    "weight_matrix.py",
)


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def parse_direct_url(text: str | None) -> dict[str, str]:
    """Extract the VCS source of an installed distribution.

    PyPI and a git checkout can both report version 0.17.0 while shipping
    different code, so the version string alone is not provenance.
    """
    try:
        payload = json.loads(text or "")
    except (TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    result: dict[str, str] = {}
    url = payload.get("url")
    if url:
        result["source"] = url
    commit = (payload.get("vcs_info") or {}).get("commit_id")
    if commit:
        result["commit"] = commit
    return result


def lalamo_root() -> Path | None:
    try:
        import lalamo

        return Path(lalamo.__file__).resolve().parent
    except Exception:  # noqa: BLE001 - fingerprint must never break a run
        return None


def lalamo_provenance() -> dict[str, str]:
    seen: dict[str, str] = {}
    try:
        seen["lalamo_version"] = version("lalamo")
        parsed = parse_direct_url(distribution("lalamo").read_text("direct_url.json"))
    except Exception:  # noqa: BLE001
        return seen
    if parsed.get("source"):
        seen["lalamo_source"] = parsed["source"]
    if parsed.get("commit"):
        seen["lalamo_commit"] = parsed["commit"]
    if not parsed:
        seen["lalamo_source"] = "pypi-or-unknown"
    return seen


def lalamo_code_hashes(root: Path | None = None) -> dict[str, str]:
    root = root or lalamo_root()
    if root is None:
        return {}
    hashes: dict[str, str] = {}
    missing = []
    for relative in FINGERPRINT_FILES:
        digest = file_sha256(root / relative)
        if digest:
            key = "lalamo_" + relative.replace("/", "_").replace(".py", "") + "_sha256"
            hashes[key] = digest
        else:
            missing.append(relative)
    hashes["lalamo_hash_complete"] = "false" if missing else "true"
    if missing:
        hashes["lalamo_hash_missing"] = ",".join(missing)
    return hashes


def fingerprint() -> dict[str, str]:
    """Everything needed to decide whether two runs used the same code."""
    seen: dict[str, str] = {"python_version": platform.python_version()}
    try:
        seen["jax_version"] = version("jax")
    except Exception:  # noqa: BLE001
        pass
    seen.update(lalamo_provenance())
    seen.update(lalamo_code_hashes())
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report = fingerprint()
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    if not any(key.endswith("_sha256") for key in report):
        print("warning: lalamo is not importable; no code hashes recorded", file=sys.stderr)


if __name__ == "__main__":
    main()
