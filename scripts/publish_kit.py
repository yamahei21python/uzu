#!/usr/bin/env python3
"""Build, publish, and *verify* the public quantization kit.

Publishing used to be three manual commands, and one of them silently gutted
the Kaggle dataset: ``kaggle datasets version`` skips directories unless
``--dir-mode zip`` is given, so the upload contained only ``kit_manifest.json``
while the kernel still "succeeded".

This script makes the failure impossible to miss:

1. build the kit from the current commit
2. upload to HF and version the Kaggle dataset (zip mode)
3. read both back and fail unless scripts/, data/, configs/ and the manifest
   commit all match the local build

Usage:
    python scripts/publish_kit.py \
        --dataset-dir /Users/kohei/Myproject/kaggleAI/datasets/uzu-quant-kit-public \
        --message "Rebuild: 8de3b348"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_PREFIXES = ("scripts/", "data/", "configs/")


def parse_manifest_commit(text: str) -> str | None:
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return None
    commit = payload.get("built_from_commit") if isinstance(payload, dict) else None
    return str(commit) if commit else None


def parse_dataset_listing_json(text: str) -> list[str]:
    """Parse ``kaggle datasets files --format json`` (a token line precedes the array)."""
    start = text.find("[")
    if start < 0:
        return []
    try:
        payload = json.loads(text[start:])
    except ValueError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item["name"]) for item in payload if isinstance(item, dict) and item.get("name")]


def parse_dataset_listing(text: str) -> list[str]:
    """Extract file names from ``kaggle datasets files`` output."""
    names: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("-") or line.lower().startswith("name "):
            continue
        name = line.split()[0]
        if name.endswith("/") or "." in name:
            names.append(name)
    return names


def verify_listing(names: list[str]) -> list[str]:
    """Return the missing pieces that would break a kernel."""
    problems = []
    for prefix in REQUIRED_PREFIXES:
        if not any(name.startswith(prefix) for name in names):
            problems.append(f"dataset has no {prefix} files")
    if not any(name == "kit_manifest.json" for name in names):
        problems.append("dataset has no kit_manifest.json")
    return problems


def run_optional(command: list[str]) -> tuple[int, str]:
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    return result.returncode, result.stdout


def fetch_kaggle_manifest(kaggle_bin: str, dataset: str, workdir: Path) -> str | None:
    result = subprocess.run(
        [kaggle_bin, "datasets", "download", "-d", dataset, "-f", "kit_manifest.json", "-p", str(workdir), "--force"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    path = workdir / "kit_manifest.json"
    return path.read_text(encoding="utf-8") if path.exists() else None


def run(command: list[str]) -> str:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.stdout.strip():
        print(result.stdout.strip()[-2000:], flush=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def fetch_text(url: str, timeout: int = 60) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8")
    except Exception:  # noqa: BLE001 - verification reports, it does not crash
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--hf-repo", default="yamahei21/uzu-quant-kit")
    parser.add_argument("--kaggle-dataset", default="yamahei21python/uzu-quant-kit-public")
    parser.add_argument("--message", default=None)
    parser.add_argument("--hf-bin", default="hf")
    parser.add_argument("--kaggle-bin", default="kaggle")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    output = args.dataset_dir
    message = args.message or "Rebuild kit"

    if not args.verify_only:
        run([sys.executable, str(ROOT / "scripts" / "build_kit.py"), "--output", str(output)])
        local_commit = parse_manifest_commit((output / "kit_manifest.json").read_text(encoding="utf-8"))
        print(f"local kit commit: {local_commit}")
        if args.dry_run:
            print("dry-run: skipping upload")
            return

        run([args.hf_bin, "upload", args.hf_repo, str(output), ".", "--repo-type", "dataset",
             "--commit-message", message])
        run([args.kaggle_bin, "datasets", "version", "-p", str(output), "-m", message,
             "--dir-mode", "zip"])

    problems: list[str] = []

    remote_manifest = fetch_text(f"https://huggingface.co/datasets/{args.hf_repo}/raw/main/kit_manifest.json")
    local_manifest = (output / "kit_manifest.json").read_text(encoding="utf-8")
    local_commit = parse_manifest_commit(local_manifest)
    remote_commit = parse_manifest_commit(remote_manifest or "")
    print(f"hf kit commit: {remote_commit}")
    if remote_commit != local_commit:
        problems.append(f"HF manifest commit {remote_commit} != local {local_commit}")

    missing: list[str] = []
    for attempt in range(4):
        code, listing = run_optional([args.kaggle_bin, "datasets", "files", args.kaggle_dataset, "--format", "json"])
        names = parse_dataset_listing_json(listing) if code == 0 else []
        if not names:
            _, listing = run_optional([args.kaggle_bin, "datasets", "files", args.kaggle_dataset])
            names = parse_dataset_listing(listing)
        missing = verify_listing(names)
        if not missing:
            break
        if attempt < 3:
            print("dataset listing incomplete; waiting for Kaggle to finish processing", flush=True)
            time.sleep(15)
    problems.extend(missing)

    with tempfile.TemporaryDirectory() as tmp:
        kaggle_manifest = fetch_kaggle_manifest(args.kaggle_bin, args.kaggle_dataset, Path(tmp))
    kaggle_commit = parse_manifest_commit(kaggle_manifest or "")
    print(f"kaggle kit commit: {kaggle_commit}")
    if kaggle_commit != local_commit:
        problems.append(f"Kaggle manifest commit {kaggle_commit} != local {local_commit}")

    if problems:
        print("\nPUBLISH INVALID")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("PUBLISH OK")


if __name__ == "__main__":
    main()
