#!/usr/bin/env python3
"""Publish lane results to the shared HF results dataset.

Results used to go to ``kentasatomuse/uzu-results``, which the owner account
cannot read, so every handoff needed a manual export. They now go to one private
dataset: ``yamahei21/uzu-results``.

The upload carries provenance because a result without it cannot be compared:

    <namespace>/environment.json     lalamo commit + RHT code hashes
    <namespace>/<lane>/manifest.json run_lane manifest (already has the fingerprint)

Uploads are refused when a manifest reports ``problems`` (a broken lane must not
look like data) or when no manifest exists at all.

Usage:
    python scripts/publish_results.py --out out --as muse
    python scripts/publish_results.py --out out --as muse --dry-run
    python scripts/publish_results.py --out out --as muse --allow-problems
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPO = "yamahei21/uzu-results"


def load_manifests(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    found = []
    for path in sorted(root.rglob("manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            print(f"skip unreadable {path}: {error}", file=sys.stderr)
            continue
        if isinstance(payload, dict):
            found.append((path, payload))
    return found


def collect_issues(root: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings).

    Errors block the upload: a lane that reported problems is not data, and an
    empty tree usually means the run wrote somewhere else.
    """
    errors: list[str] = []
    warnings: list[str] = []
    manifests = load_manifests(root)
    if not manifests:
        errors.append(f"no manifest.json under {root}")
        return errors, warnings

    for path, manifest in manifests:
        lane = manifest.get("lane") or path.parent.name
        problems = manifest.get("problems") or []
        if problems:
            errors.append(f"{lane}: {len(problems)} problem(s), first: {problems[0]}")
        fingerprint = manifest.get("environment_fingerprint") or {}
        if not fingerprint:
            warnings.append(f"{lane}: no environment_fingerprint (older kit?)")
        elif fingerprint.get("lalamo_hash_complete") == "false":
            warnings.append(f"{lane}: lalamo fingerprints incomplete; do not compare with owner")
    return errors, warnings


def result_files(root: Path) -> list[str]:
    """Relative paths that are worth uploading (skip caches and temp output)."""
    skipped = {".git", "__pycache__", ".ipynb_checkpoints"}
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if any(part in skipped for part in relative.parts):
            continue
        files.append(str(relative))
    return files


def write_environment(root: Path) -> Path:
    """Record which code produced these numbers, next to the numbers."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from env_fingerprint import fingerprint
    except Exception as error:  # noqa: BLE001 - provenance is best effort
        print(f"warning: cannot compute fingerprint: {error}", file=sys.stderr)
        return root
    path = root / "environment.json"
    path.write_text(json.dumps(fingerprint(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def upload(repo: str, root: Path, namespace: str) -> None:
    command = [
        "hf",
        "upload",
        repo,
        str(root),
        namespace,
        "--repo-type",
        "dataset",
        "--commit-message",
        f"results: {namespace}",
    ]
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"hf upload failed ({result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="lane output directory")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--as", dest="namespace", required=True, help="subdirectory, e.g. muse or owner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-problems", action="store_true")
    args = parser.parse_args()

    if not args.out.exists():
        raise SystemExit(f"nothing at {args.out}")

    errors, warnings = collect_issues(args.out)
    for warning in warnings:
        print(f"warning: {warning}")
    if errors and not args.allow_problems:
        print("REFUSING TO PUBLISH")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    for error in errors:
        print(f"overridden: {error}")

    files = result_files(args.out)
    print(f"{len(files)} file(s) under {args.out}")
    if args.dry_run:
        print(f"dry-run: would upload to {args.repo}/{args.namespace}")
        return

    environment = write_environment(args.out)
    print(f"wrote {environment}")
    upload(args.repo, args.out, args.namespace)
    print(f"PUBLISHED {args.repo}/{args.namespace}")


if __name__ == "__main__":
    main()
