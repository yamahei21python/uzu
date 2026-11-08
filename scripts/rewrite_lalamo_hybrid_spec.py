#!/usr/bin/env python3
"""Rewrite lalamo HybridSpec metadata in a safetensors file to uzu's flat form.

lalamo:
  {"type":"HybridSpec", ..., "incoherence_spec":{"type":"HadamardSpec","block_size":32,"processing_mode":"input_output"}}
uzu:
  {"type":"HybridSpec", ..., "incoherence_block_size":32, "incoherence_processing_mode":"input_output"}

Only the safetensors header metadata is rewritten. Tensor bytes are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


def _parse_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError("not a safetensors file: short header length")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_bytes = f.read(header_len)
        if len(header_bytes) != header_len:
            raise ValueError("not a safetensors file: truncated header")
    return json.loads(header_bytes), 8 + header_len


def _flatten_hybrid(spec: dict) -> dict:
    if spec.get("type") != "HybridSpec":
        return spec
    if "incoherence_spec" in spec:
        incoherence = spec.pop("incoherence_spec")
        if incoherence is None:
            spec["incoherence_block_size"] = None
            spec["incoherence_processing_mode"] = "input_output"
        else:
            spec["incoherence_block_size"] = incoherence.get("block_size")
            spec["incoherence_processing_mode"] = incoherence.get("processing_mode", "input_output")
    else:
        spec.setdefault("incoherence_block_size", None)
        spec.setdefault("incoherence_processing_mode", "input_output")
    return spec


def rewrite(path: Path, *, dry_run: bool = False, backup: bool = True) -> int:
    header, data_start = _parse_header(path)
    metadata = header.get("__metadata__")
    if not metadata:
        print("no __metadata__ section")
        return 0

    changed = 0
    for key, value in list(metadata.items()):
        if not key.endswith(".spec") and key != "spec":
            continue
        try:
            spec = json.loads(value)
        except json.JSONDecodeError:
            continue
        if spec.get("type") != "HybridSpec":
            continue
        new_spec = _flatten_hybrid(spec)
        metadata[key] = json.dumps(new_spec, separators=(",", ":"))
        changed += 1
        print(f"rewrite {key}")
        print(f"  old: {value}")
        print(f"  new: {metadata[key]}")

    if changed == 0:
        print("no HybridSpec metadata found")
        return 0

    if dry_run:
        print(f"dry-run: would rewrite {changed} metadata entries")
        return changed

    new_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (8 - (len(new_header) % 8)) % 8
    new_header += b" " * padding

    tmp = path.with_name(path.name + ".rewrite.tmp")
    backup_path = path.with_name(path.name + ".lalamo.bak")
    if backup:
        # Always refresh: a leftover backup from an earlier run describes
        # different weights, and eval copies prefer the backup.
        shutil.copy2(path, backup_path)
        print(f"backup: {backup_path}")

    with path.open("rb") as src, tmp.open("wb") as dst:
        dst.write(struct.pack("<Q", len(new_header)))
        dst.write(new_header)
        src.seek(data_start)
        shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)
    os.replace(tmp, path)
    if backup:
        # The backup is written before the rewrite, so its mtime is always
        # older than the rewritten file. Record which file it belongs to.
        stamp = path.with_name(path.name + ".lalamo.bak.stamp")
        stamp.write_text(str(path.stat().st_mtime_ns), encoding="utf-8")
    print(f"rewrote {changed} metadata entries in {path}")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()
    rewrite(args.path, dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == "__main__":
    main()
